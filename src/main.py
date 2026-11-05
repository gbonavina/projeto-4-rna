"""Keyword spotting on Google Speech Commands v0.02 (35 words) with a BiLSTM.

Pipeline: official speaker-disjoint split -> log-mel features (GPU) ->
train with augmentation (train only) -> pick best epoch on VALIDATION ->
evaluate the test set exactly once.

Follow a run:   tail -f outputs/run.log   (or outputs/$RUN_NAME/run.log)
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LSTMClassifier
from speech_data import CLIP, N_MELS, LogMel, list_splits, load_noise, load_split

CFG = dict(
    hidden_size=256,
    num_layers=2,
    dropout=0.2,
    bidirectional=True,
    batch_size=256,
    lr=2e-3,
    weight_decay=1e-2,
    epochs=40,
    label_smoothing=0.1,
    # augmentation (train only)
    shift_ms=100,
    noise_prob=0.8,
    # "absolute": noise gain ~ U(0, noise_max_vol) of full scale (final model).
    # "snr": noise mixed relative to speech level at U(snr_db) dB (ablation,
    #        94.5% test vs 95.1%, not selected on validation).
    noise_mode="absolute",
    noise_max_vol=0.1,
    snr_db=(5.0, 20.0),
    freq_masks=2, freq_mask_width=8,
    time_masks=2, time_mask_width=12,
    seed=0,
)
RUN_NAME = os.environ.get("RUN_NAME", "")
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / RUN_NAME


def log(msg: str = "") -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Frontend(nn.Module):
    """waveform -> normalised log-mel, with train-only augmentation."""

    def __init__(self, noise: torch.Tensor, cfg: dict):
        super().__init__()
        self.logmel = LogMel()
        self.register_buffer("noise", noise)
        self.register_buffer("mean", torch.zeros(N_MELS))
        self.register_buffer("std", torch.ones(N_MELS))
        self.cfg = cfg

    @torch.no_grad()
    def fit_stats(self, train_wav: torch.Tensor) -> None:
        """Per-mel-band mean/std from TRAIN clips only (clean, unaugmented)."""
        s, s2, n = 0.0, 0.0, 0
        for i in range(0, len(train_wav), 2048):
            m = self.logmel(to_float(train_wav[i : i + 2048]))
            s = s + m.sum(dim=(0, 1))
            s2 = s2 + (m**2).sum(dim=(0, 1))
            n += m.shape[0] * m.shape[1]
        self.mean = s / n
        self.std = (s2 / n - self.mean**2).sqrt()

    def augment_wave(self, wav: torch.Tensor) -> torch.Tensor:
        B, dev = wav.shape[0], wav.device
        ar = torch.arange(CLIP, device=dev)
        # random time shift with zero fill
        m = int(self.cfg["shift_ms"] * CLIP / 1000)
        shift = torch.randint(-m, m + 1, (B,), device=dev)
        wav = F.pad(wav, (m, m)).gather(1, (m - shift)[:, None] + ar[None, :])
        # Mix in a random slice of background noise.
        start = torch.randint(0, len(self.noise) - CLIP, (B,), device=dev)
        seg = self.noise[start[:, None] + ar[None, :]]
        keep = torch.rand(B, device=dev) < self.cfg["noise_prob"]
        if self.cfg.get("noise_mode", "absolute") == "absolute":
            vol = torch.rand(B, device=dev) * self.cfg["noise_max_vol"] * keep
            return (wav + vol[:, None] * seg).clamp(-1, 1)
        # "snr" mode. Speech power = mean power of the loudest 25 frames
        # (0.25 s), so silence around the word doesn't lower the estimate.
        frame_pow = wav.unfold(1, 160, 160).pow(2).mean(dim=2)  # (B, 100)
        p_speech = frame_pow.topk(25, dim=1).values.mean(dim=1)
        p_noise = seg.pow(2).mean(dim=1).clamp_min(1e-10)
        lo, hi = self.cfg["snr_db"]
        snr = lo + (hi - lo) * torch.rand(B, device=dev)
        scale = torch.sqrt(p_speech / (p_noise * 10 ** (snr / 10)))
        scale = scale * keep
        return (wav + scale[:, None] * seg).clamp(-1, 1)

    def spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        B, T, Fq = x.shape
        dev = x.device
        for _ in range(self.cfg["freq_masks"]):
            w = torch.randint(0, self.cfg["freq_mask_width"] + 1, (B, 1, 1), device=dev)
            f0 = (torch.rand(B, 1, 1, device=dev) * (Fq - w)).long()
            f = torch.arange(Fq, device=dev)[None, None, :]
            x = x.masked_fill((f >= f0) & (f < f0 + w), 0.0)
        for _ in range(self.cfg["time_masks"]):
            w = torch.randint(0, self.cfg["time_mask_width"] + 1, (B, 1, 1), device=dev)
            t0 = (torch.rand(B, 1, 1, device=dev) * (T - w)).long()
            t = torch.arange(T, device=dev)[None, :, None]
            x = x.masked_fill((t >= t0) & (t < t0 + w), 0.0)
        return x

    def forward(self, wav: torch.Tensor, train: bool) -> torch.Tensor:
        wav = to_float(wav)
        if train:
            wav = self.augment_wave(wav)
        x = (self.logmel(wav) - self.mean) / self.std
        return self.spec_augment(x) if train else x


def to_float(wav_int16: torch.Tensor) -> torch.Tensor:
    return wav_int16.float() / 32768.0


def batches(wav, y, batch_size, shuffle):
    idx = torch.randperm(len(y), device=y.device) if shuffle else torch.arange(len(y), device=y.device)
    for i in range(0, len(y), batch_size):
        b = idx[i : i + batch_size]
        yield wav[b], y[b]


@torch.no_grad()
def predict(model, front, wav, y, batch_size=1024) -> np.ndarray:
    model.eval()
    preds = [model(front(w, train=False)).argmax(1) for w, _ in batches(wav, y, batch_size, False)]
    return torch.cat(preds).cpu().numpy()


def main() -> None:
    cfg = CFG
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    t0 = time.time()
    classes, files = list_splits()
    data = {name: load_split(name, f, classes) for name, f in files.items()}
    log(f"{len(classes)} classes | " + "  ".join(
        f"{k}: {len(v.y)} clips" for k, v in data.items())
        + f" | speaker-disjoint split verified | loaded in {time.time() - t0:.0f}s")

    tensors = {k: (torch.from_numpy(v.wav).to(device), torch.from_numpy(v.y).to(device))
               for k, v in data.items()}
    front = Frontend(torch.from_numpy(load_noise()).to(device), cfg).to(device)
    front.fit_stats(tensors["train"][0])

    model = LSTMClassifier(N_MELS, cfg["hidden_size"], cfg["num_layers"], len(classes),
                           cfg["dropout"], cfg["bidirectional"]).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * -(-len(tensors["train"][1]) // cfg["batch_size"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, cfg["lr"], total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    hist = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    wav_tr, y_tr = tensors["train"]
    log("")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t_ep, total, correct, n = time.time(), 0.0, 0, 0
        for w, y in batches(wav_tr, y_tr, cfg["batch_size"], shuffle=True):
            logits = model(front(w, train=True))
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        val_pred = predict(model, front, *tensors["val"])
        val_acc = (val_pred == data["val"].y).mean()
        hist["train_loss"].append(total / n)
        hist["train_acc"].append(correct / n)
        hist["val_acc"].append(float(val_acc))
        star = ""
        if val_acc > best_acc:
            best_acc, best_state, best_epoch, star = val_acc, copy.deepcopy(model.state_dict()), epoch, "  *best"
        log(f"epoch {epoch:2d}/{cfg['epochs']}  loss {total / n:.4f}  train acc {correct / n:.4f}  "
            f"val acc {val_acc:.4f}  lr {scheduler.get_last_lr()[0]:.1e}  "
            f"{time.time() - t_ep:.0f}s{star}")

    model.load_state_dict(best_state)
    log(f"\nBest epoch {best_epoch} (val acc {best_acc:.4f}), chosen on validation only.")

    # --- one-time test evaluation ---
    y_test = data["test"].y
    test_pred = predict(model, front, *tensors["test"])
    test_acc = (test_pred == y_test).mean()
    log(f"\nTEST accuracy: {test_acc:.4f}  ({(test_pred == y_test).sum()}/{len(y_test)})"
        f"   chance: {1 / len(classes):.4f}")

    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (y_test, test_pred), 1)
    per_class = cm.diagonal() / cm.sum(axis=1)
    order = np.argsort(per_class)
    log("\nHardest words (test accuracy):")
    for i in order[:6]:
        top_wrong = np.argsort(cm[i] * (np.arange(len(classes)) != i))[::-1][0]
        log(f"  {classes[i]:9s} {per_class[i]:.3f}   most often confused with '{classes[top_wrong]}'"
            f" ({cm[i, top_wrong]}x)")
    log("Easiest: " + ", ".join(f"{classes[i]} {per_class[i]:.3f}" for i in order[-5:][::-1]))

    (OUT_DIR / "results.json").write_text(json.dumps({
        "config": cfg, "best_epoch": best_epoch, "val_acc": float(best_acc),
        "test_acc": float(test_acc),
        "per_class_test_acc": {c: float(a) for c, a in zip(classes, per_class)},
    }, indent=2))
    plot(hist, cm, classes, best_epoch)
    torch.save({"model": model.state_dict(), "frontend": front.state_dict(),
                "classes": classes, "config": cfg}, OUT_DIR / "bilstm.pt")
    log(f"\nSaved results.json, plots and bilstm.pt to {OUT_DIR}")


def plot(hist, cm, classes, best_epoch) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = np.arange(1, len(hist["val_acc"]) + 1)
    ax.plot(ep, hist["train_acc"], label="train (augmented)")
    ax.plot(ep, hist["val_acc"], label="val (clean)")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best epoch {best_epoch}")
    ax.set(xlabel="epoch", ylabel="accuracy", title="Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy.png", dpi=120)

    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="predicted", ylabel="actual", title="Test confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()
