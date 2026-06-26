from pathlib import Path

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


class EnCodecWrapper:
    def __init__(self, bandwidth=6.0, device=None):
        self.bandwidth = bandwidth
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(self.device)
        self.model.eval()

        self.sample_rate = self.model.sample_rate
        self.channels = self.model.channels

    def load_audio(self, wav_path):
        wav_path = Path(wav_path)

        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        wav, sr = torchaudio.load(str(wav_path))

        wav = convert_audio(
            wav,
            sr,
            self.sample_rate,
            self.channels
        )

        wav = wav.unsqueeze(0)
        wav = wav.to(self.device)

        return wav

    @torch.no_grad()
    def encode_waveform(self, wav):
        encoded_frames = self.model.encode(wav)

        codes = torch.cat(
            [frame[0] for frame in encoded_frames],
            dim=-1
        )

        scales = [frame[1] for frame in encoded_frames]

        return codes, scales, encoded_frames

    @torch.no_grad()
    def decode_frames(self, encoded_frames):
        reconstructed = self.model.decode(encoded_frames)
        return reconstructed

    @torch.no_grad()
    def encode_file(self, wav_path):
        wav = self.load_audio(wav_path)
        codes, scales, encoded_frames = self.encode_waveform(wav)

        return {
            "wav": wav,
            "codes": codes,
            "scales": scales,
            "encoded_frames": encoded_frames,
        }

    @torch.no_grad()
    def reconstruct_file(self, wav_path, output_path):
        result = self.encode_file(wav_path)
        reconstructed = self.decode_frames(result["encoded_frames"])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        wav_out = reconstructed.squeeze(0).cpu()

        torchaudio.save(
            str(output_path),
            wav_out,
            self.sample_rate
        )

        return {
            "input_path": str(wav_path),
            "output_path": str(output_path),
            "codes_shape": tuple(result["codes"].shape),
            "sample_rate": self.sample_rate,
            "bandwidth": self.bandwidth,
        }


if __name__ == "__main__":
    wrapper = EnCodecWrapper(bandwidth=6.0)

    print("EnCodec 24 kHz model loaded.")
    print(f"Device      : {wrapper.device}")
    print(f"Sample rate : {wrapper.sample_rate}")
    print(f"Channels    : {wrapper.channels}")
    print(f"Bandwidth   : {wrapper.bandwidth} kbps")
