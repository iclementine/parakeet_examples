import time
from pathlib import Path
import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import parakeet
from parakeet.data import dataset
from parakeet.models.waveflow import UpsampleNet, WaveFlow, ConditionalWaveFlow, WaveFlowLoss
from parakeet.audio import AudioProcessor
from parakeet.utils import scheduler
from parakeet.training.cli import default_argument_parser
from parakeet.utils.mp_tools import rank_zero_only

from config import get_cfg_defaults
from ljspeech import LJSpeech, LJSpeechClipCollector, LJSpeechCollector


def main_sp(config, args):
    if args.nprocs > 1 and args.device=="gpu":
        dist.init_parallel_env()
    ljspeech_dataset = LJSpeech(args.data)
    valid_set, train_set = dataset.split(ljspeech_dataset, config.data.valid_size)

    batch_fn = LJSpeechClipCollector(config.data.clip_frames, config.data.hop_length)
    
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

    valid_batch_fn = LJSpeechCollector()
    valid_loader = DataLoader(
        valid_set, batch_size=config.data.batch_size, collate_fn=valid_batch_fn)

    encoder = UpsampleNet(config.model.upsample_factors)
    decoder = WaveFlow(
        n_flows=config.model.n_flows,
        n_layers=config.model.n_layers,
        n_group=config.model.n_group,
        channels=config.model.channels,
        mel_bands=config.data.n_mels,
        kernel_size=config.model.kernel_size,
    )
    model = ConditionalWaveFlow(encoder, decoder)

    if args.nprocs > 1:
        model = paddle.DataParallel(model)
    optimizer = paddle.optimizer.Adam(2e-4, parameters=model.parameters())
    criterion = WaveFlowLoss(config.model.sigma)

    if dist.get_rank() == 0:
        visualizer = SummaryWriter(logdir=args.output)
        output_dir = visualizer.logdir
        checkpoint_dir = Path(output_dir) / "checkpoints"
    else:
        visualizer, output_dir, checkpoint_dir = None, None, None

    iteration = 0
    iterator = iter(train_loader)

    def compute_outputs(mel, wav):
        # model_core = model._layers if isinstance(model, paddle.DataParallel) else model
        z, log_det_jocobian = model(wav, mel)
        return z, log_det_jocobian

    def compute_losses(inputs, outputs):
        loss = criterion(outputs)
        return loss
    
    @rank_zero_only
    def save():
        parakeet.utils.io.save_parameters(
            str(checkpoint_dir), iteration, model, optimizer)
    
    def load():
        loaded_iteration = parakeet.utils.io.load_parameters(
            model, optimizer, checkpoint_dir=str(checkpoint_dir))
        nonlocal iteration
        iteration = loaded_iteration
    
    # @rank_zero_only
    # def valid():
    #     valid_iterator = iter(valid_loader)
    #     valid_losses = defaultdict(list)
    #     text, mel, stop_label = next(valid_iterator)
    #     with paddle.no_grad():
    #         outputs = compute_outputs(text, mel, stop_label)
    #         losses = compute_losses((text, mel, stop_label), outputs)
    #         for k, v in losses.items():
    #             valid_losses[k].append(float(v))
    #     valid_losses = {k: np.mean(v) for k, v in valid_losses.items()}
    #     for k, v in valid_losses.items():
    #         visualizer.add_scalar(f"valid/{k}", v, iteration)
    
    @rank_zero_only
    def log_states():
        visualizer.add_scalar(f"train_loss/loss", float(loss), iteration)
    
    load()
    while iteration <= config.training.max_iteration:
        iteration += 1
        try:
            mel, wav= next(iterator)
        except:
            iterator = iter(train_loader)
            mel, wav = next(iterator)
        optimizer.clear_grad()
        model.train()
        outputs = compute_outputs(mel, wav)
        loss = compute_losses((mel, wav), outputs)
        print(float(loss))
        loss.backward() 
        optimizer.step()

        # other stuffs
        log_states()

        # if iteration % config.training.valid_interval == 0:
        #     valid()
        
        if iteration % config.training.save_interval == 0:
            save()

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
