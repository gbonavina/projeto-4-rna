"""Plot example clips: waveform, log-mel (model input), and augmented log-mel.

Usage:  uv run python src/plot_audio.py [word ...]
        uv run python src/plot_audio.py yes no tree three

Saves outputs/audio_examples.png and a .wav copy of each example so
you can listen to exactly what is plotted.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

from main import CFG, OUT_DIR, Frontend, to_float
from speech_data import CACHE_DIR, HOP, SR, list_splits, load_noise



def main(words: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    classes, splits = list_splits()
    wav = np.load(CACHE_DIR / "train_wav.npy", mmap_mode="r")
    y = np.load(CACHE_DIR / "train_y.npy")
    words = [w for w in words if w in classes] or ["yes", "no", "tree", "three"]

    torch.manual_seed(0)
    front = Frontend(torch.from_numpy(load_noise()), CFG)
    # Normalisation stats from a train subset, just for display.
    rng = np.random.default_rng(0)
    front.fit_stats(torch.from_numpy(wav[np.sort(rng.choice(len(y), 2000, replace=False))]))

    fig, axes = plt.subplots(3, len(words), figsize=(4.2 * len(words), 9), squeeze=False)
    t_wave = np.arange(SR) / SR
    for col, word in enumerate(words):
        i = int(rng.choice(np.flatnonzero(y == classes.index(word))))
        clip = torch.from_numpy(np.array(wav[i : i + 1]))
        clean = front(clip, train=False)[0].numpy()
        aug = front(clip, train=True)[0].numpy()
        wavfile.write(OUT_DIR / f"example_{word}.wav", SR, clip[0].numpy())

        ax = axes[0, col]
        ax.plot(t_wave, to_float(clip)[0].numpy(), lw=0.5)
        ax.set(title=f"'{word}'  ({splits['train'][i]})", xlabel="time (s)",
               ylabel="amplitude" if col == 0 else None, ylim=(-1, 1))
        ax.title.set_fontsize(9)

        extent = [0, clean.shape[0] * HOP / SR, 0, clean.shape[1]]
        vmin, vmax = np.percentile(clean, [1, 99])
        for row, (spec, label) in enumerate([(clean, "log-mel (model input)"),
                                             (aug, "augmented (training only)")], 1):
            ax = axes[row, col]
            ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma",
                      extent=extent, vmin=vmin, vmax=vmax)
            ax.set(xlabel="time (s)", ylabel=f"{label}\nmel band" if col == 0 else None)

    fig.suptitle("Speech Commands examples: waveform -> log-mel spectrogram "
                 "(101 frames x 40 mel bands = one LSTM input sequence)")
    fig.tight_layout()
    path = OUT_DIR / "audio_examples.png"
    fig.savefig(path, dpi=110)
    print(f"Saved {path} and example_<word>.wav files")


if __name__ == "__main__":
    main(sys.argv[1:])
