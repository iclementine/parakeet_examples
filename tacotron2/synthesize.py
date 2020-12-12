import argparse
from pathlib import Path
import numpy as np

import paddle
import parakeet
from parakeet.frontend import EnglishCharacter
from parakeet.models.tacotron2 import Tacotron2

from config import get_cfg_defaults

def main(config, args):
    paddle.set_device(args.device)

    # model
    frontend = EnglishCharacter()
    model = Tacotron2.from_pretrained(frontend, config, args.checkpoint_path)
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

    for i, sentence in enumerate(sentences):
        mel_output, _ = model.predict(sentence)
        mel_output = mel_output.T 

        np.save(str(output_dir / f"sentence_{i}"), mel_output)
        if args.verbose:
            print("spectrogram saved at {}".format(output_dir / f"sentence_{i}.npy"))

if __name__ == "__main__":
    config = get_cfg_defaults()

    parser = argparse.ArgumentParser(description="generate mel spectrogram with TransformerTTS.")
    parser.add_argument("--config", type=str, metavar="FILE", help="extra config to overwrite the default config")
    parser.add_argument("--checkpoint_path", type=str, help="path of the checkpoint to load.")
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
