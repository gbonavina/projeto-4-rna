"""Report figures for one trained run.

Usage:  uv run python src/report_plots.py [run_folder]
        (default: outputs/, the final model chosen on validation)

Writes to <run_folder>/report/:
  01_dataset.png             clips per word in each split
  02_training_curves.png     training loss + train/val accuracy per epoch
  03_confusion_matrix.png    row-normalised test confusion matrix
  04_top_confusions.png      the most frequent (true -> predicted) mistakes
  05_per_class_metrics.png   precision / recall / F1 per word
  06_confidence.png          confidence of right vs wrong answers + calibration
  07_noise_robustness.png    test accuracy when noise is added at a given SNR
  08_embeddings_tsne.png     2-D map of what the LSTM learned
  09_misclassified.png       log-mels of mistakes, with true/predicted word
  per_class_metrics.csv      the numbers behind figure 05
  summary.json               headline numbers
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support

from main import Frontend, batches, to_float
from model import LSTMClassifier
from speech_data import CLIP, HOP, N_MELS, SR, list_splits, load_noise, load_split

ROOT = Path(__file__).resolve().parent.parent

# Palette (validated reference palette, light mode)
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    "axes.axisbelow": True,
})


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"  saved {name}")


def load_run(run: Path, device):
    ck = torch.load(run / "bilstm.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    model = LSTMClassifier(N_MELS, cfg["hidden_size"], cfg["num_layers"], len(ck["classes"]),
                           cfg["dropout"], cfg["bidirectional"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    front = Frontend(torch.from_numpy(load_noise()), cfg).to(device)
    front.load_state_dict(ck["frontend"])  # train-set normalisation stats
    return model, front, ck["classes"]


@torch.no_grad()
def infer(model, front, wav, y, batch=1024):
    """Returns softmax probabilities and pooled embeddings (input to the head)."""
    probs, embs = [], []
    for w, _ in batches(wav, y, batch, shuffle=False):
        x = front(w, train=False)
        out, _ = model.lstm(model.input_norm(x))
        pooled = torch.cat([out.mean(1), out.amax(1)], dim=1)
        hidden = model.head[:3](pooled)  # dropout(no-op in eval) -> linear -> GELU
        probs.append(torch.softmax(model.head[3:](hidden), dim=1).cpu())
        embs.append(hidden.cpu())
    return torch.cat(probs).numpy(), torch.cat(embs).numpy()


def add_noise(wav_i16: torch.Tensor, noise: torch.Tensor, snr_db: float, gen) -> torch.Tensor:
    """Test-time noise at a fixed SNR (speech power = loudest 0.25 s)."""
    wav = to_float(wav_i16)
    B, dev = wav.shape[0], wav.device
    start = torch.randint(0, len(noise) - CLIP, (B,), generator=gen).to(dev)
    seg = noise[start[:, None] + torch.arange(CLIP, device=dev)[None, :]]
    p_speech = wav.unfold(1, 160, 160).pow(2).mean(2).topk(25, dim=1).values.mean(1)
    p_noise = seg.pow(2).mean(1).clamp_min(1e-10)
    scale = torch.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return ((wav + scale[:, None] * seg).clamp(-1, 1) * 32767).to(torch.int16)


def parse_log(run: Path) -> dict:
    pat = re.compile(r"epoch\s+(\d+)/\d+\s+loss ([\d.]+)\s+train acc ([\d.]+)\s+val acc ([\d.]+)")
    rows = [tuple(map(float, m.groups())) for m in pat.finditer((run / "run.log").read_text())]
    ep, loss, tr, va = map(np.array, zip(*rows))
    return {"epoch": ep, "loss": loss, "train_acc": tr, "val_acc": va}


# ------------------------------------------------------------------ figures
def fig_dataset(data, classes, out):
    counts = {k: np.bincount(v.y, minlength=len(classes)) for k, v in data.items()}
    order = np.argsort(counts["train"])[::-1]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(classes))
    w = 0.28
    for i, (split, color) in enumerate([("train", BLUE), ("val", ORANGE), ("test", AQUA)]):
        ax.bar(x + (i - 1) * w, counts[split][order], w - 0.03, color=color, label=split)
    ax.set_xticks(x, [classes[i] for i in order], rotation=60, ha="right")
    ax.set(ylabel="clips", title="Clips per word in each split (speaker-disjoint)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=3, loc="upper right")
    save(fig, out, "01_dataset.png")


def fig_training(hist, best_epoch, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(hist["epoch"], hist["loss"], color=BLUE)
    a1.set(xlabel="epoch", ylabel="cross-entropy (label smoothing 0.1)",
           title="Training loss (augmented clips)")
    a2.plot(hist["epoch"], hist["train_acc"], color=BLUE, label="train (augmented)")
    a2.plot(hist["epoch"], hist["val_acc"], color=ORANGE, label="validation (clean)")
    for ax in (a1, a2):
        ax.axvline(best_epoch, color=MUTED, ls="--", lw=1)
    a2.annotate(f"best epoch {best_epoch}\nval {hist['val_acc'][best_epoch - 1]:.1%}",
                (best_epoch, hist["val_acc"][best_epoch - 1]), xytext=(-95, -45),
                textcoords="offset points", color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    a2.set(xlabel="epoch", ylabel="accuracy", ylim=(0, 1), title="Accuracy per epoch")
    a2.legend(loc="lower right")
    save(fig, out, "02_training_curves.png")


def fig_confusion(cm, classes, out):
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.grid(False)
    ax.set(xlabel="predicted word", ylabel="true word",
           title="Test confusion matrix (each row sums to 100%)")
    # label the notable off-diagonal cells so small errors are readable
    for i, j in zip(*np.nonzero((norm >= 0.02) & ~np.eye(len(classes), dtype=bool))):
        ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=6.5, color=INK)
    cb = fig.colorbar(im, fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, out, "03_confusion_matrix.png")


def fig_top_confusions(cm, classes, out, k=15):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:k], off.shape))[0]
    labels = [f"{classes[i]} → {classes[j]}" for i, j in idx]
    vals = [off[i, j] for i, j in idx]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(k)[::-1]
    ax.barh(y, vals, color=BLUE, height=0.7)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, str(v), va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="test clips", title=f"Top {k} mistakes (true → predicted)")
    save(fig, out, "04_top_confusions.png")


def fig_per_class(y, pred, classes, out):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
    with open(out / "per_class_metrics.csv", "w") as fh:
        fh.write("word,precision,recall,f1,support\n")
        for c, *vals in zip(classes, p, r, f, s):
            fh.write(f"{c},{vals[0]:.4f},{vals[1]:.4f},{vals[2]:.4f},{vals[3]}\n")
    order = np.argsort(f)
    fig, ax = plt.subplots(figsize=(8, 10))
    yy = np.arange(len(classes))
    for vals, color, mk, label in [(p, BLUE, "o", "precision"), (r, ORANGE, "s", "recall"),
                                   (f, INK, "|", "F1")]:
        ax.scatter(vals[order], yy, color=color, marker=mk, s=60 if mk != "|" else 220,
                   label=label, zorder=3, edgecolors=SURFACE if mk != "|" else None,
                   linewidths=1.5 if mk != "|" else 2)
    ax.hlines(yy, np.minimum(p, r)[order], np.maximum(p, r)[order], color=GRID, lw=2, zorder=2)
    ax.set_yticks(yy, [classes[i] for i in order])
    ax.set(xlabel="score", xlim=(0.75, 1.0), title="Precision, recall and F1 per word (test)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncols=3)
    save(fig, out, "05_per_class_metrics.png")
    return float(f.mean())


def fig_confidence(probs, y, out, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    right = pred == y
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = np.linspace(0, 1, 41)
    a1.hist(conf[right], bins=bins, color=BLUE, histtype="step", lw=2,
            label=f"correct ({right.sum()})")
    a1.hist(conf[~right], bins=bins, color=ORANGE, histtype="step", lw=2,
            label=f"wrong ({(~right).sum()})")
    a1.set_yscale("log")
    a1.set(xlabel="model confidence (max softmax probability)", ylabel="test clips (log scale)",
           title="Confidence: right vs wrong answers")
    a1.legend(loc="upper left")

    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    acc = np.array([right[ids == b].mean() if (ids == b).any() else np.nan for b in range(n_bins)])
    mid = np.array([conf[ids == b].mean() if (ids == b).any() else np.nan for b in range(n_bins)])
    cnt = np.bincount(ids, minlength=n_bins)
    ece = np.nansum(cnt / len(y) * np.abs(acc - mid))
    a2.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="perfectly calibrated")
    ok = ~np.isnan(acc)
    a2.plot(mid[ok], acc[ok], color=BLUE, marker="o", ms=8, mec=SURFACE, mew=2, label="model")
    a2.set(xlabel="mean confidence in bin", ylabel="actual accuracy in bin", xlim=(0, 1), ylim=(0, 1),
           title=f"Calibration (ECE = {ece:.3f})")
    a2.legend(loc="upper left")
    save(fig, out, "06_confidence.png")
    return float(ece)


def fig_noise(model, front, wav, y, noise, out):
    snrs = [-5, 0, 5, 10, 20]
    gen = torch.Generator().manual_seed(0)
    accs = []
    for snr in snrs:
        noisy = torch.cat([add_noise(w, noise, snr, gen) for w, _ in batches(wav, y, 2048, False)])
        p, _ = infer(model, front, noisy, y)
        accs.append(float((p.argmax(1) == y.cpu().numpy()).mean()))
    p, _ = infer(model, front, wav, y)
    clean = float((p.argmax(1) == y.cpu().numpy()).mean())
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(snrs, accs, color=BLUE, marker="o", ms=8, mec=SURFACE, mew=2, label="noisy test clips")
    ax.axhline(clean, color=MUTED, ls="--", lw=1)
    ax.text(snrs[0], clean + 0.012, f"clean test: {clean:.1%}", color=INK2)
    for s, a in zip(snrs, accs):
        ax.annotate(f"{a:.1%}", (s, a), xytext=(0, -16), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9)
    ax.set(xlabel="signal-to-noise ratio (dB)  ← noisier", ylabel="test accuracy", ylim=(0, 1),
           xticks=snrs, title="Robustness: test accuracy with background noise added")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower right")
    save(fig, out, "07_noise_robustness.png")
    return dict(zip(map(str, snrs), accs))


def fig_tsne(emb, y, classes, out, per_class=80):
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class, (y == c).sum()),
                                     replace=False) for c in range(len(classes))])
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb[idx])
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(xy[:, 0], xy[:, 1], s=8, color=BLUE, alpha=0.35, linewidths=0)
    for c in range(len(classes)):
        pts = xy[y[idx] == c]
        cx, cy = np.median(pts, axis=0)
        ax.text(cx, cy, classes[c], ha="center", va="center", fontsize=9, color=INK,
                weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    ax.set(xticks=[], yticks=[],
           title=f"t-SNE of the LSTM's learned representation ({per_class} test clips per word)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, out, "08_embeddings_tsne.png")


def fig_misclassified(front, wav, y, probs, classes, out, n=12):
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.flatnonzero(pred != y)
    pick = wrong[np.argsort(conf[wrong])[::-1][:n]]  # the most confident mistakes
    with torch.no_grad():
        specs = front(wav[torch.from_numpy(pick).to(wav.device)], train=False).cpu().numpy()
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    extent = [0, specs.shape[1] * HOP / SR, 0, N_MELS]
    for ax, s, i in zip(axes.flat, specs, pick):
        ax.imshow(s.T, origin="lower", aspect="auto", cmap="magma", extent=extent)
        ax.set_title(f"true '{classes[y[i]]}' → predicted '{classes[pred[i]]}' ({conf[i]:.0%})",
                     fontsize=9, loc="center")
        ax.grid(False)
        ax.set(xlabel="time (s)", ylabel="mel band")
    fig.suptitle("Most confident mistakes on the test set (log-mel input)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, out, "09_misclassified.png")


def main(run: Path) -> None:
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run}\nDevice: {device}")

    classes_now, files = list_splits()
    data = {k: load_split(k, f, classes_now) for k, f in files.items()}
    model, front, classes = load_run(run, device)
    assert classes == classes_now
    results = json.loads((run / "results.json").read_text())

    wav = torch.from_numpy(data["test"].wav).to(device)
    y_t = torch.from_numpy(data["test"].y).to(device)
    y = data["test"].y
    probs, emb = infer(model, front, wav, y_t)
    pred = probs.argmax(1)
    acc = float((pred == y).mean())
    assert abs(acc - results["test_acc"]) < 1e-3, "checkpoint doesn't reproduce logged test acc"
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)

    fig_dataset(data, classes, out)
    fig_training(parse_log(run), results["best_epoch"], out)
    fig_confusion(cm, classes, out)
    fig_top_confusions(cm, classes, out)
    macro_f1 = fig_per_class(y, pred, classes, out)
    ece = fig_confidence(probs, y, out)
    noise_acc = fig_noise(model, front, wav, y_t, torch.from_numpy(load_noise()).to(device), out)
    fig_tsne(emb, y, classes, out)
    fig_misclassified(front, wav, y, probs, classes, out)

    summary = {"test_accuracy": acc, "val_accuracy": results["val_acc"],
               "best_epoch": results["best_epoch"], "macro_f1": macro_f1, "ece": ece,
               "test_accuracy_by_snr_db": noise_acc}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    default = ROOT / "outputs"
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default)
