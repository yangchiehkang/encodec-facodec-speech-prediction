import argparse
from pathlib import Path

import torchaudio

from codec.encodec_wrapper import EnCodecWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/recon.wav")
    parser.add_argument("--bandwidth", type=float, default=6.0)
    args = parser.parse_args()

    wrapper = EnCodecWrapper(bandwidth=args.bandwidth)

    result = wrapper.encode_file(args.wav)

    input_wav = result["wav"]
    codes = result["codes"]
    encoded_frames = result["encoded_frames"]

    reconstructed = wrapper.decode_frames(encoded_frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recon_wav = reconstructed.squeeze(0).cpu()

    torchaudio.save(
        str(out_path),
        recon_wav,
        wrapper.sample_rate
    )

    print(f"Input wav shape: {input_wav.squeeze(0).shape}")
    print(f"Token shape: {codes.squeeze(0).shape}")
    print(f"Reconstructed wav shape: {recon_wav.shape}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
