import time
import logging
from pathlib import Path
import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import parakeet
from parakeet.data import dataset
from parakeet.frontend import English
from parakeet.models.transformer_tts import TransformerTTS, TransformerTTSLoss, AdaptiveTransformerTTSLoss
from parakeet.utils import scheduler, checkpoint, mp_tools, display
from parakeet.training.cli import default_argument_parser

from config import get_cfg_defaults
from ljspeech import LJSpeech, LJSpeechCollector, Transform

def main_sp(config, args):
    if args.nprocs > 1 and args.device=="gpu":
        dist.init_parallel_env()

    ljspeech_dataset = LJSpeech(args.data)
    transform = Transform(config.data.mel_start_value, config.data.mel_end_value)
    ljspeech_dataset = dataset.TransformDataset(ljspeech_dataset, transform)
    valid_set, train_set = dataset.split(ljspeech_dataset, config.data.valid_size)
    batch_fn = LJSpeechCollector(padding_idx=config.data.padding_idx)
    
    if args.nprocs == 1:
        train_loader = DataLoader(
            train_set, 
            batch_size=config.data.batch_size, 
            shuffle=True, 
            drop_last=True,
            collate_fn=batch_fn)
    else:
        sampler = DistributedBatchSampler(
            train_set, 
            batch_size=config.data.batch_size,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True)
        train_loader = DataLoader(
            train_set, batch_sampler=sampler, collate_fn=batch_fn)

    valid_loader = DataLoader(
        valid_set, batch_size=config.data.batch_size, collate_fn=batch_fn)

    frontend = English()
    model = TransformerTTS(
        frontend, 
        d_encoder=config.model.d_encoder,
        d_decoder=config.model.d_decoder,
        d_mel=config.data.d_mel,
        n_heads=config.model.n_heads,
        d_ffn=config.model.d_ffn,
        encoder_layers=config.model.encoder_layers,
        decoder_layers=config.model.decoder_layers,
        d_prenet=config.model.d_prenet,
        d_postnet=config.model.d_postnet,
        postnet_layers=config.model.postnet_layers,
        postnet_kernel_size=config.model.postnet_kernel_size,
        max_reduction_factor=config.model.max_reduction_factor,
        decoder_prenet_dropout=config.model.decoder_prenet_dropout,
        dropout=config.model.dropout)
    if args.nprocs > 1:
        model = paddle.DataParallel(model)
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-9,
        parameters=model.parameters()
    )
    criterion = TransformerTTSLoss(config.model.stop_loss_scale)
    drop_n_heads = scheduler.StepWise(config.training.drop_n_heads)
    reduction_factor = scheduler.StepWise(config.training.reduction_factor)
    
    checkpoint_path = args.checkpoint_path
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    logger.addHandler(logging.StreamHandler())
    log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
    logger.addHandler(logging.FileHandler(str(log_file)))
    if dist.get_rank() == 0:
        visualizer = SummaryWriter(logdir=str(output_dir))

    iteration = 0
    epoch = 0

    epoch += 1
    if args.nprocs > 1 and args.device=="gpu":
        train_loader.batch_sampler.set_epoch(epoch)
    iterator = iter(train_loader)

    def compute_outputs(text, mel, stop_label):
        model_core = model._layers if isinstance(model, paddle.DataParallel) else model
        model_core.set_constants(
            reduction_factor(iteration), drop_n_heads(iteration))

        # TODO(chenfeiyu): we can combine these 2 slices
        mel_input = mel[:,:-1, :]
        reduced_mel_input = mel_input[:, ::model_core.r, :]
        outputs = model(text, reduced_mel_input)
        return outputs

    def compute_losses(inputs, outputs):
        _, mel, stop_label = inputs
        mel_target = mel[:, 1:, :]
        stop_label_target = stop_label[:, 1:]

        mel_output = outputs["mel_output"]
        mel_intermediate = outputs["mel_intermediate"]
        stop_logits = outputs["stop_logits"]

        time_steps = mel_target.shape[1]
        losses = criterion(
            mel_output[:,:time_steps, :], 
            mel_intermediate[:,:time_steps, :], 
            mel_target, 
            stop_logits[:,:time_steps, :], 
            stop_label_target)
        return losses

    @mp_tools.rank_zero_only
    @paddle.fluid.dygraph.no_grad
    def valid(data_loader):
        valid_losses = defaultdict(list)
        for i, batch in enumerate(data_loader):
            text, mel, stop_label = batch
            outputs = compute_outputs(text, mel, stop_label)
            losses = compute_losses((text, mel, stop_label), outputs)
            for k, v in losses.items():
                valid_losses[k].append(float(v))
            
            if i < 2:
                attention_weights = outputs["cross_attention_weights"]
                display.add_attention_plots(
                    visualizer, 
                    f"valid_sentence_{i}_cross_attention_weights", 
                    attention_weights, 
                    iteration)
        # write visual log
        valid_losses = {k: np.mean(v) for k, v in valid_losses.items()}
        for k, v in valid_losses.items():
            visualizer.add_scalar(f"valid/{k}", v, iteration)


    @mp_tools.rank_zero_only
    def log_states(losses):
        for k, v in losses.items():
            visualizer.add_scalar(f"train_loss/{k}", v, iteration)
    
    # resume or load
    iteration = checkpoint.load_parameters(
        model, 
        optimizer, 
        checkpoint_dir=str(checkpoint_dir), 
        checkpoint_path=checkpoint_path)
    
    while iteration <= config.training.max_iteration:
        iteration += 1
        start = time.time()
        try:
            text, mel, stop_label = next(iterator)
        except StopIteration:
            # new epoch
            epoch += 1
            if args.nprocs > 1 and args.device=="gpu":
                train_loader.batch_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            text, mel, stop_label = next(iterator)
        data_loader_time = time.time() - start

        optimizer.clear_grad()
        model.train()
        outputs = compute_outputs(text, mel, stop_label)
        losses = compute_losses((text, mel, stop_label), outputs)
        loss = losses["loss"]
        loss.backward() 
        optimizer.step()
        iteration_time = time.time() - start

        losses_np = {k: float(v) for k, v in losses.items()}
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time, iteration_time)
        msg += ', '.join('{}: {:>.6f}'.format(k, v) for k, v in losses_np.items())
        # print(msg)
        logger.info(msg)
        log_states(losses_np)

        if iteration % config.training.valid_interval == 0:
            valid(valid_loader)
        
        if iteration % config.training.save_interval == 0:
            checkpoint.save_parameters(str(checkpoint_dir), iteration, model, optimizer)

def main(config, args):
    if args.nprocs > 1 and args.device == "gpu":
        dist.spawn(main_sp, args=(config, args), nprocs=args.nprocs)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.config: 
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
