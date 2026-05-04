"""
Train a small CNN on the synthetic shape dataset, then render the
dashboard figures used in README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from generate_data import CLASS_NAMES, DataConfig, generate, save_preview

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 11,
})


# ---------------------------------------------------------------- model ----
class SmallCNN(nn.Module):
    """
    Three conv blocks (16 → 32 → 64 channels) with max-pooling between them,
    then a flatten and a two-layer MLP head. Input 64×64×3 → 8×8×64 after
    the conv stack, flattened to 4096, classified to 3 classes.
    """

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 64 → 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 32 → 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 16 → 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                          # 8*8*64 = 4096
            nn.Linear(8 * 8 * 64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -------------------------------------------------------------- training ---
def to_tensor(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    Xt = torch.from_numpy(X).float().permute(0, 3, 1, 2) / 255.0
    yt = torch.from_numpy(y).long()
    return Xt, yt


def train(epochs: int = 15, batch_size: int = 32, lr: float = 5e-4,
          seed: int = 42) -> dict:
    torch.manual_seed(seed)

    cfg = DataConfig()
    X_train, y_train, X_test, y_test = generate(cfg)

    Xt, yt = to_tensor(X_train, y_train)
    Xv, yv = to_tensor(X_test, y_test)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size)

    model = SmallCNN(n_classes=3)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += float(loss) * yb.size(0)
            correct += int((logits.argmax(1) == yb).sum())
            tot += yb.size(0)
        tr_loss, tr_acc = loss_sum / tot, correct / tot

        model.eval()
        tot_v, correct_v, loss_sum_v = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                loss_sum_v += float(F.cross_entropy(logits, yb)) * yb.size(0)
                correct_v += int((logits.argmax(1) == yb).sum())
                tot_v += yb.size(0)
        te_loss, te_acc = loss_sum_v / tot_v, correct_v / tot_v

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(f"epoch {epoch:2d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  |  "
              f"test loss {te_loss:.3f} acc {te_acc:.3f}")

    # Final test predictions for confusion matrix and gallery.
    model.eval()
    with torch.no_grad():
        all_logits = model(Xv)
        y_pred = all_logits.argmax(1).numpy()

    return {
        "history": history,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
        "X_train": X_train,
        "y_train": y_train,
    }


# ---------------------------------------------------------------- figures --
def fig_training_curves(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], color=COLOR_BLUE, marker="o",
                 linewidth=1.6, label="train")
    axes[0].plot(epochs, history["test_loss"], color=COLOR_RED, marker="o",
                 linewidth=1.6, label="test")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss curves"); axes[0].legend(loc="upper right")

    axes[1].plot(epochs, history["train_acc"], color=COLOR_BLUE, marker="o",
                 linewidth=1.6, label="train")
    axes[1].plot(epochs, history["test_acc"], color=COLOR_RED, marker="o",
                 linewidth=1.6, label="test")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curves"); axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right")

    fig.suptitle("Training dynamics — small CNN on synthetic shapes",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    n_per = cm.sum(axis=1, keepdims=True)
    norm = cm / n_per
    cmap = LinearSegmentedColormap.from_list("blue_only", ["#FFFFFF", COLOR_BLUE])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            color = "white" if norm[i, j] > 0.4 else COLOR_TEXT
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.2%})",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=color)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"pred {n}" for n in CLASS_NAMES])
    ax.set_yticklabels([f"true {n}" for n in CLASS_NAMES])
    ax.set_title("Confusion matrix on the test set")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalized rate")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_misclassified(X_test, y_test, y_pred, out_path: Path) -> None:
    wrong = np.where(y_test != y_pred)[0]
    if len(wrong) == 0:
        # Make a "perfect classification" placeholder so the README still has 5 figs.
        fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "No misclassifications on this run.",
                ha="center", va="center", fontsize=14, color=COLOR_TEXT)
        ax.axis("off")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    pick = np.random.default_rng(0).choice(wrong, size=min(8, len(wrong)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), constrained_layout=True)
    for ax, idx in zip(axes.ravel(), pick):
        ax.imshow(X_test[idx])
        ax.set_title(f"true: {CLASS_NAMES[y_test[idx]]}\npred: {CLASS_NAMES[y_pred[idx]]}",
                     fontsize=10, color=COLOR_RED)
        ax.axis("off")
    # Hide any unused axes.
    for k in range(len(pick), len(axes.ravel())):
        axes.ravel()[k].axis("off")
    fig.suptitle(f"Misclassified test samples ({len(wrong)} total wrong out of {len(y_test)})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    out = train(epochs=15)

    Path("results").mkdir(exist_ok=True)
    summary = {
        "epochs": 15,
        "final_train_acc": float(out["history"]["train_acc"][-1]),
        "final_test_acc": float(out["history"]["test_acc"][-1]),
        "history": {k: [float(v) for v in vs] for k, vs in out["history"].items()},
        "n_test": int(len(out["y_test"])),
        "n_test_wrong": int((out["y_test"] != out["y_pred"]).sum()),
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    assets = Path("assets"); assets.mkdir(exist_ok=True)

    # Sample image grid (uses a freshly generated mini-set so the preview is
    # deterministic regardless of which split it draws from).
    cfg_preview = DataConfig(seed=123)
    X_prev, y_prev, _, _ = generate(cfg_preview)
    save_preview(assets / "01_samples.png", X_prev, y_prev, cfg_preview)

    fig_training_curves(out["history"], assets / "02_curves.png")
    fig_confusion(out["y_test"], out["y_pred"], assets / "03_confusion.png")
    fig_misclassified(out["X_test"], out["y_test"], out["y_pred"],
                      assets / "04_misclassified.png")

    print(f"\nFinal test accuracy: {summary['final_test_acc']:.3f}")
    print(f"Figures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
