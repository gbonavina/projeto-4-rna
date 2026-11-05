# Keyword spotting with a bidirectional LSTM

Classifies 1-second spoken commands from **Google Speech Commands v0.02** into
35 words ("yes", "no", "up", "tree", "three", ...) using log-mel spectrograms
and a 2-layer bidirectional LSTM (PyTorch).

**Test accuracy: 95.1%** (35 classes, chance = 2.9%) on speakers never seen in
training.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync                                  # create .venv and install dependencies
uv run python src/download_data.py       # fetch the dataset (~2.4 GB download, ~3.8 GB extracted)
```

The dataset is **not stored in this repository** (it is larger than GitHub's
file limits). `download_data.py` downloads it from Google's official URL into
`data/speech_commands/`, verifies that all 105,835 `.wav` files are present,
and deletes the archive. It needs ~7 GB of free disk space while running.
Re-running it is a no-op once the data is complete; use `--force` to start
over or `--keep-archive` to keep the `.tar.gz`.

## Usage

```bash
# Train (≈11 min on one GPU; runs on CPU too, slowly). Writes to outputs/ (or outputs/$RUN_NAME/)
# Follow with: tail -f outputs/run.log
uv run python src/main.py
RUN_NAME=my_run uv run python src/main.py   # keep a separate run

# Report figures -> <run folder>/report/   (default: outputs/)
uv run python src/report_plots.py
uv run python src/report_plots.py outputs/my_run

# Plot waveform / log-mel / augmented log-mel for some words
uv run python src/plot_audio.py yes no tree three
```

Model checkpoints (`*.pt`) are not committed; `main.py` regenerates them.

## Project layout

| File | Purpose |
|---|---|
| `src/download_data.py` | Download, extract and verify the dataset |
| `src/speech_data.py` | Official splits, speaker-disjointness check, waveform cache, log-mel features |
| `src/model.py` | `LSTMClassifier`: BiLSTM + mean/max pooling over time + MLP head |
| `src/main.py` | Training (augmentation, model selection on validation, one-time test evaluation) |
| `src/report_plots.py` | Report figures: curves, confusion matrix, per-class metrics, calibration, noise robustness, t-SNE, mistakes |
| `src/plot_audio.py` | Visualise example clips |
| `outputs/` | Final model: `run.log`, `results.json`, training plots, `report/` figures |
| `PROJECT_LOG.md` | Full record of the work, decisions and results (for writing the report) |

## Method

- **Features:** 1 s at 16 kHz → 25 ms Hann windows, 10 ms hop → 512-point FFT →
  40 mel bands → log, giving a 101 × 40 sequence per clip. Per-band
  normalisation statistics come from the training set only.
- **Model:** 2-layer BiLSTM (256 units per direction, dropout 0.2), mean + max
  pooling over time, MLP head → 35 logits (2.7M parameters). Cross-entropy with
  label smoothing 0.1 (softmax is inside the loss).
- **Training:** AdamW, one-cycle LR (peak 2e-3), 40 epochs, batch 256.
  Augmentation on training batches only: ±100 ms time shift, background noise
  from the dataset's `_background_noise_` recordings (80% of clips, gain up to
  0.1 of full scale), SpecAugment (2 frequency + 2 time masks).
- **Model selection:** the epoch with the best **validation** accuracy; the
  test set is evaluated once, at the end.

## No data leakage

- Uses the official `validation_list.txt` / `testing_list.txt`, which assign
  files by a hash of the speaker id. `speech_data.py` asserts that no file and
  no speaker appears in more than one split (2,112 / 256 / 250 speakers).
- Normalisation statistics: train only. Augmentation: train only.
  Background-noise recordings contain no speech.
- Hyperparameters and the best epoch are chosen on validation; test is never
  used for any decision.

## Results (final model, `outputs/`)

| Metric | Value |
|---|---|
| Validation accuracy (best epoch 36) | 96.0% |
| **Test accuracy** | **95.1%** |
| Macro F1 (test) | 0.948 |
| Expected calibration error | 0.085 (under-confident, due to label smoothing) |
| Test accuracy with added noise at 20 / 10 / 5 / 0 dB SNR | 93.7% / 87.0% / 77.6% / 61.1% |

Most errors are between similar-sounding words: off→up, tree↔three,
four↔forward↔follow, learn→left.

## Dataset

P. Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech
Recognition*, 2018, [arXiv:1804.03209](https://arxiv.org/abs/1804.03209).
Licensed under CC BY 4.0.
