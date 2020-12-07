import argparse
import time
from pathlib import Path
import numpy as np
import paddle

import parakeet
from parakeet.frontend import English
from parakeet.models.transformer_tts import TransformerTTS
from parakeet.utils import scheduler
from parakeet.training.cli import default_argument_parser
from parakeet.utils.display import add_attention_plots

from config import get_cfg_defaults

@paddle.fluid.dygraph.no_grad
def main(config, args):
    paddle.set_device(args.device)

    # model
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
    drop_n_heads = scheduler.StepWise(config.training.drop_n_heads)
    reduction_factor = scheduler.StepWise(config.training.reduction_factor)
    
    # load latest checkpoint from checkpoint_dir 
    # or load a specified checkpoint from checkpoint_path
    iteration = parakeet.utils.checkpoint.load_parameters(
        model, 
        checkpoint_dir=args.checkpoint_dir, 
        checkpoint_path=args.checkpoint_path)
    model.set_constants(
        reduction_factor=reduction_factor(iteration), 
        drop_n_heads=drop_n_heads(iteration))
    model.eval()

    # inputs
    input_path = Path(args.input).expanduser()
    with open(input_path, "rt") as f: 
        sentences = f.readlines()
    
    if args.output is None:
        output_dir = input_path.parent / "synthesis"
    else:
        output_dir = Path(args.output).expanduser()
    output_dir.mkdir(exist_ok=True)

    with paddle.no_grad():
        for i, sentence in enumerate(sentences):
            outputs = model.predict(sentence, verbose=args.verbose)
            mel_output = outputs["mel_output"]
            # cross_attention_weights = outputs["cross_attention_weights"]
            mel_output = paddle.transpose(mel_output, [0, 2, 1]).numpy()[0] #(C, T)
            np.save(str(output_dir / f"sentence_{i}"), mel_output)
            if args.verbose:
                print("spectrogram saved at {}".format(output_dir / f"sentence_{i}.npy"))

if __name__ == "__main__":
    config = get_cfg_defaults()

    parser = argparse.ArgumentParser(description="generate mel spectrogram with TransformerTTS.")
    parser.add_argument("--config", type=str, metavar="FILE", help="extra config to overwrite the default config")
    parser.add_argument("--checkpoint_path", type=str, help="path of the checkpoint to load.")
    parser.add_argument("--checkpoint_dir", type=str, help="path from which to load the latest checkpoint.")
    parser.add_argument("--input", type=str, help="path of the text sentences")
    parser.add_argument("--output", type=str, help="path to save outputs")
    parser.add_argument("--device", type=str, default="cpu", help="device type to use.")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="options to overwrite --config file and the default config, passing in KEY VALUE pairs")
    parser.add_argument("-v", "--verbose", action="store_true", help="print msg")
    
    args = parser.parse_args()
    if args.config: 
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
