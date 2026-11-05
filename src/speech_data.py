"""Google Speech Commands v0.02: file lists, waveform cache, log-mel features.

Leakage guarantees (checked by `check_no_leakage`):
1. Official split lists (validation_list.txt / testing_list.txt). Google
   assigns files by hashing the SPEAKER id, so all clips of one person land
   in the same split. We assert that no speaker appears in two splits and no
   file is in two splits.
2. Feature normalisation statistics are computed on TRAIN clips only.
3. Augmentation (time shift, background noise, SpecAugment) is applied to
   TRAIN batches only. Background noise comes from `_background_noise_`,
   which contains no speech, so it carries no speaker/label information.
4. Val/test are scored on the clean, unaugmented clips.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = ROOT / "speech_commands"
CACHE_DIR = ROOT / "speech_cache"
SR = 16000
CLIP = SR  # every clip is padded/cropped to 1 s
N_MELS = 40
N_FFT, WIN, HOP = 512, 400, 160  # 25 ms window, 10 ms hop -> 101 frames


@dataclass
class SplitData:
    wav: np.ndarray  # (n, 16000) int16
    y: np.ndarray  # (n,) int64 class index
    files: list[str]


def speaker_of(rel_path: str) -> str:
    """'yes/0a7c2a8d_nohash_0.wav' -> '0a7c2a8d'"""
    return rel_path.split("/")[1].split("_nohash_")[0]


def list_splits() -> tuple[list[str], dict[str, list[str]]]:
    if not (RAW_DIR / "testing_list.txt").exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DIR}. Run:  uv run python src/download_data.py")
    classes = sorted(p.name for p in RAW_DIR.iterdir()
                     if p.is_dir() and not p.name.startswith("_"))
    val = set((RAW_DIR / "validation_list.txt").read_text().split())
    test = set((RAW_DIR / "testing_list.txt").read_text().split())
    all_files = sorted(f"{c}/{p.name}" for c in classes
                       for p in (RAW_DIR / c).glob("*.wav"))
    splits = {
        "train": [f for f in all_files if f not in val and f not in test],
        "val": [f for f in all_files if f in val],
        "test": [f for f in all_files if f in test],
    }
    check_no_leakage(splits)
    return classes, splits


def check_no_leakage(splits: dict[str, list[str]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            fa, fb = set(splits[a]), set(splits[b])
            assert not fa & fb, f"files shared between {a} and {b}"
            sa = {speaker_of(f) for f in fa}
            sb = {speaker_of(f) for f in fb}
            shared = sa & sb
            assert not shared, f"{len(shared)} speakers shared between {a} and {b}"


def _read(path: Path) -> np.ndarray:
    sr, x = wavfile.read(path)
    assert sr == SR and x.dtype == np.int16, path
    out = np.zeros(CLIP, dtype=np.int16)
    out[: min(len(x), CLIP)] = x[:CLIP]  # zero-pad short clips at the end
    return out


def load_split(name: str, files: list[str], classes: list[str]) -> SplitData:
    """Waveforms are cached as .npy so the ~105k wav files are read once."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wav_path, y_path = CACHE_DIR / f"{name}_wav.npy", CACHE_DIR / f"{name}_y.npy"
    if not wav_path.exists():
        idx = {c: i for i, c in enumerate(classes)}
        wav = np.stack([_read(RAW_DIR / f) for f in files])
        y = np.array([idx[f.split("/")[0]] for f in files], dtype=np.int64)
        np.save(wav_path, wav)
        np.save(y_path, y)
    return SplitData(np.load(wav_path), np.load(y_path), files)


def load_noise() -> np.ndarray:
    """All background-noise recordings concatenated, as float32 in [-1, 1]."""
    parts = []
    for p in sorted((RAW_DIR / "_background_noise_").glob("*.wav")):
        sr, x = wavfile.read(p)
        if sr == SR:
            parts.append(x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0))
    return np.concatenate(parts)


# ---------------------------------------------------------------- features
def mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, sr=SR, fmin=20.0, fmax=SR / 2) -> np.ndarray:
    """Triangular HTK-style mel filters, shape (n_fft//2 + 1, n_mels)."""
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((len(freqs), n_mels))
    for m in range(n_mels):
        lo, mid, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[:, m] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb.astype(np.float32)


class LogMel(torch.nn.Module):
    """waveform (B, 16000) float -> log-mel (B, 101, 40)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wav, N_FFT, HOP, WIN, self.window, center=True,
                          return_complex=True)  # (B, freq, frames)
        power = spec.abs().pow(2).transpose(1, 2)  # (B, frames, freq)
        return torch.log(power @ self.fb + 1e-6)


if __name__ == "__main__":
    classes, splits = list_splits()
    print(f"{len(classes)} classes: {classes}")
    for name, files in splits.items():
        spk = {speaker_of(f) for f in files}
        print(f"{name:5s} {len(files):6d} clips, {len(spk):5d} speakers")
    print("Leakage checks passed (no shared files or speakers).")
