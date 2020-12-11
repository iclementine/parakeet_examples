import argparse
import numpy as np
import soundfile as sf
import os
from pathlib import Path
import paddle
import parakeet
from parakeet.models.waveflow import UpsampleNet, WaveFlow, ConditionalWaveFlow
from parakeet.utils import layer_tools, checkpoint


from config import get_cfg_defaults

def main(config, args):
    paddle.set_device(args.device)
    encoder = UpsampleNet(config.model.upsample_factors)
    decoder = WaveFlow(
        n_flows=config.model.n_flows,
        n_layers=config.model.n_layers,
        n_group=config.model.n_group,
        channels=config.model.channels,
        mel_bands=config.data.n_mels,
        kernel_size=config.model.kernel_size)
    model = ConditionalWaveFlow(encoder, decoder)
    checkpoint.load_parameters(
        model, 
        checkpoint_dir=args.checkpoint_dir, 
        checkpoint_path=args.checkpoint_path)

    for layer in model.sublayers():
        try:
            paddle.nn.utils.remove_weight_norm(layer)
        except:
            pass
    layer_tools.summary(model)
    model.eval()

    mel_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in mel_dir.iterdir():
        mel = np.load(str(file_path))
        mel = paddle.to_tensor(mel)
        mel = paddle.unsqueeze(mel, 0)
        audio = model.synthesize(mel)
        audio = audio.numpy()[0]
        audio_path = output_dir / (os.path.splitext(file_path.name)[0] + ".wav")
        sf.write(audio_path, audio, config.data.sample_rate)
        print("[synthesize] {} -> {}".format(file_path, audio_path))


if __name__ == "__main__":
    config = get_cfg_defaults()

    parser = argparse.ArgumentParser(description="generate mel spectrogram with TransformerTTS.")
    parser.add_argument("--config", type=str, metavar="FILE", help="extra config to overwrite the default config")
    parser.add_argument("--checkpoint_path", type=str, help="path of the checkpoint to load.")
    parser.add_argument("--checkpoint_dir", type=str, help="path from which to load the latest checkpoint.")
    parser.add_argument("--input", type=str, help="path of directory containing mel spectrogram (in .npy format)")
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