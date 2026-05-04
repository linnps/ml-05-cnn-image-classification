"""
Synthetic shape-classification dataset, rendered from scratch.

Three classes — circle, square, triangle — each drawn at a random
position, size, color, and rotation onto a noisy coloured background.
Difficulty knobs let us produce an "easy" set for first training and
a "hard" set for ablation studies.

Why synthetic?
- Zero copyright concerns (every byte is original).
- Difficulty is parameterized; we can make a "hard" variant by raising
  noise / shrinking shapes / increasing background-foreground colour
  overlap.
- Class balance is exactly controlled; no long-tail surprises.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["circle", "square", "triangle"]


@dataclass
class DataConfig:
    image_size: int = 64
    n_per_class_train: int = 800
    n_per_class_test: int = 150
    bg_noise_std: float = 6.0
    shape_size_min: float = 0.30    # as fraction of image_size
    shape_size_max: float = 0.50
    seed: int = 42


def _shape_color(rng: np.random.Generator) -> tuple[int, int, int]:
    # Light foreground colors so the shapes pop against the dark backgrounds.
    return tuple(int(c) for c in rng.integers(190, 245, size=3))


def _bg(rng: np.random.Generator, size: int, noise_std: float) -> Image.Image:
    # Dark bluish-grey backgrounds keep contrast high while staying within
    # the portfolio palette (no fluorescent colours).
    base = rng.integers(15, 70, size=3)
    arr = np.full((size, size, 3), base, dtype=np.float32)
    arr += rng.normal(0, noise_std, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_circle(draw: ImageDraw.ImageDraw, cx, cy, r, color) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def _draw_square(draw: ImageDraw.ImageDraw, cx, cy, r, color, angle) -> None:
    side = r * 1.7
    pts = np.array([[-side / 2, -side / 2], [side / 2, -side / 2],
                    [side / 2, side / 2], [-side / 2, side / 2]])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    pts = pts @ rot.T + np.array([cx, cy])
    draw.polygon([tuple(p) for p in pts], fill=color)


def _draw_triangle(draw: ImageDraw.ImageDraw, cx, cy, r, color, angle) -> None:
    h = r * np.sqrt(3) * 0.9
    pts = np.array([[0, -h * 2 / 3], [-r, h / 3], [r, h / 3]])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    pts = pts @ rot.T + np.array([cx, cy])
    draw.polygon([tuple(p) for p in pts], fill=color)


def _render_one(class_idx: int, cfg: DataConfig, rng: np.random.Generator) -> np.ndarray:
    img = _bg(rng, cfg.image_size, cfg.bg_noise_std)
    draw = ImageDraw.Draw(img)

    s = cfg.image_size
    r = int(rng.uniform(cfg.shape_size_min, cfg.shape_size_max) * s / 2)
    margin = r + 2
    cx = int(rng.integers(margin, s - margin))
    cy = int(rng.integers(margin, s - margin))
    # Constrain rotation so squares and triangles still look like squares
    # and triangles (small rotations only — the network shouldn't have to
    # solve full rotational invariance to learn shape category).
    angle = float(rng.uniform(-np.pi / 8, np.pi / 8))
    color = _shape_color(rng)

    if class_idx == 0:
        _draw_circle(draw, cx, cy, r, color)
    elif class_idx == 1:
        _draw_square(draw, cx, cy, r, color, angle)
    else:
        _draw_triangle(draw, cx, cy, r, color, angle)

    return np.array(img, dtype=np.uint8)


def generate(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    def make(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for cls in range(3):
            for _ in range(n_per_class):
                Xs.append(_render_one(cls, cfg, rng))
                ys.append(cls)
        idx = rng.permutation(len(ys))
        return np.stack(Xs)[idx], np.array(ys)[idx]

    X_train, y_train = make(cfg.n_per_class_train)
    X_test, y_test = make(cfg.n_per_class_test)
    return X_train, y_train, X_test, y_test


def save_preview(out_path: Path, X: np.ndarray, y: np.ndarray, cfg: DataConfig) -> None:
    """Save a 3×6 preview grid (one row per class)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 6, figsize=(11, 5.5), constrained_layout=True)
    for cls in range(3):
        idx = np.where(y == cls)[0][:6]
        for ax, i in zip(axes[cls], idx):
            ax.imshow(X[i]); ax.axis("off")
        axes[cls, 0].set_ylabel(CLASS_NAMES[cls], fontsize=11, fontweight="bold")
        axes[cls, 0].axis("on")
        axes[cls, 0].set_xticks([]); axes[cls, 0].set_yticks([])
        axes[cls, 0].spines["top"].set_visible(False)
        axes[cls, 0].spines["right"].set_visible(False)
        axes[cls, 0].spines["bottom"].set_visible(False)
        axes[cls, 0].spines["left"].set_visible(False)
    fig.suptitle("Synthetic shapes — random size / position / colour / rotation, noisy backgrounds",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic shape-classification dataset.")
    p.add_argument("--n-per-class-train", type=int, default=500)
    p.add_argument("--n-per-class-test", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = DataConfig(
        n_per_class_train=args.n_per_class_train,
        n_per_class_test=args.n_per_class_test,
        seed=args.seed,
    )
    X_train, y_train, X_test, y_test = generate(cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "shapes.npz",
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Saved to: {args.out_dir / 'shapes.npz'}")


if __name__ == "__main__":
    main()
