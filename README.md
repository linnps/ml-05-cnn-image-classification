<div align="center">

# CNN Image Classification — Shapes from Scratch in PyTorch

**A small CNN learns to tell circles, squares, and triangles apart on synthetic shape images.**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![framework](https://img.shields.io/badge/framework-PyTorch-3B6EA8?style=flat-square)
![data](https://img.shields.io/badge/data-self--generated-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> Render a 3-class shape dataset from scratch with PIL — circles, squares, triangles, randomly placed and lightly rotated on noisy dark backgrounds — then train a small CNN end-to-end in PyTorch. Watch loss collapse and test accuracy climb toward 100% in a handful of epochs.

<p align="center">
  <img src="https://img.shields.io/badge/Final_test_accuracy-99.6%25-3B6EA8?style=for-the-badge" alt="Final test accuracy 99.6%">
  <img src="https://img.shields.io/badge/Final_train_accuracy-100%25-3B6EA8?style=for-the-badge" alt="Final train accuracy 100%">
  <img src="https://img.shields.io/badge/Test_errors-2_/_450-C04040?style=for-the-badge" alt="Test errors 2 / 450">
</p>
<p align="center"><sub>Test accuracy &rarr; <b>99.6%</b> (448 / 450 correct)&nbsp;·&nbsp;Train accuracy &rarr; <b>100%</b> at epoch 15&nbsp;·&nbsp;Only <b>2</b> test images misclassified out of 450</sub></p>

<sub>**Headline finding:** a small CNN (three conv layers, no batch norm, no augmentation) is enough to push past 99% on this task in well under a minute on CPU. The two test images that *do* get misclassified are visually ambiguous (a small square at the edge looks rounded; a circle near a square pose) — exactly the kind of error we'd hope a working model leaves behind.</sub>

---

## Experimental setup

Everything below is fixed by `seed = 42` and reproduces deterministically on any machine with the pinned library versions.

### Data-generating process

The dataset is entirely synthetic — no external download. `generate_data.py` renders each image from scratch using PIL:

1. **Background.** A 64×64 RGB canvas is filled with a dark base color sampled uniformly from `[15, 70)` per channel, then Gaussian noise with `std = 6.0` is added pixel-wise and clipped to `[0, 255]`. This gives low-saturation, noisy dark backgrounds.
2. **Shape.** A foreground color is sampled uniformly from `[190, 245)` per channel (near-white), giving high contrast against the dark background. The shape radius is drawn uniformly from 30%–50% of the half-image-size (i.e. an effective radius of 9–16 px). Center position is chosen uniformly with a margin equal to the radius. Rotation angle is drawn uniformly from `[−π/8, π/8]` (±22.5°).
3. **Classes and how each is drawn:**
   - **Circle** (class 0): `ImageDraw.ellipse` centered at `(cx, cy)` with radius `r`. No rotation applied.
   - **Square** (class 1): A rotated quadrilateral with side ≈ `1.7 × r`, drawn as a polygon after applying a 2D rotation matrix.
   - **Triangle** (class 2): An equilateral-ish triangle with apex at top, height ≈ `0.87 × r × √3`, similarly rotated.
4. **Dataset sizes.** 800 images per class for training (2 400 total), 150 per class for test (450 total). Both splits are generated deterministically in one `numpy.random.default_rng(42)` call — train first, then test, with a `rng.permutation` shuffle applied to each split independently.

| Parameter | Value | Why |
|---|---|---|
| `image_size` | 64 | Small enough for fast CPU training; large enough to resolve shape geometry clearly. |
| `n_per_class_train` | 800 | 2 400 training images total; comfortably larger than the model's parameter count (~540 K). |
| `n_per_class_test` | 150 | 450 test images total; 150 per class gives reliable per-class estimates. |
| `bg_noise_std` | 6.0 | Mild noise — enough to prevent identical pixels but not enough to obscure shapes. |
| `shape_size_min / max` | 0.30 / 0.50 | Shapes occupy 30–50 % of the half-image; always visible, never trivially small. |
| `rotation range` | ±π/8 (±22.5°) | Enough variation to prevent a rotation-invariance shortcut; not so much that squares look circular. |
| `seed` | 42 | One seed drives the entire generation — background noise, positions, colors, and the permutation shuffle. |

### Preprocessing

- **Normalization.** Raw `uint8` pixel values in `[0, 255]` are divided by `255.0` in `to_tensor()`, mapping all inputs to `[0, 1]`. This is done inline before the DataLoader, with no per-channel mean/std normalization applied.
- **No data augmentation.** Variability (random position, color, rotation, noise) is baked into the dataset at generation time. The training DataLoader uses `shuffle=True` but applies no transforms.

### CNN architecture

```
Input (3, 64, 64)
  │
  ├─ Conv2d(3→16,  3×3, pad=1) → ReLU
  ├─ Conv2d(16→16, 3×3, pad=1) → ReLU
  ├─ MaxPool2d(2×2)                          →  (16, 32, 32)
  │
  ├─ Conv2d(16→32, 3×3, pad=1) → ReLU
  ├─ MaxPool2d(2×2)                          →  (32, 16, 16)
  │
  ├─ Conv2d(32→64, 3×3, pad=1) → ReLU
  ├─ MaxPool2d(2×2)                          →  (64,  8,  8)
  │
  └─ Flatten (4 096) → Linear(4096→128) → ReLU → Linear(128→3)
```

The first block has two successive 3×3 conv layers before pooling; blocks 2 and 3 each have one. No BatchNorm, no Dropout. (~540 K trainable parameters total.)

### Training hyperparameters

| Knob | Value | Why |
|---|---|---|
| Optimizer | Adam, `lr = 5e-4` | Stable adaptive optimizer; no LR schedule or warmup needed at this scale. |
| Batch size | 32 | ~75 batches per epoch; small batches provide gradient noise that acts as implicit regularization. |
| Loss | Cross-entropy (`F.cross_entropy`) | Standard multiclass classification loss; operates on raw logits. |
| Epochs | 15 | Convergence is effectively reached by epoch ≈ 8; the tail verifies stability. |
| Weight initialization | PyTorch defaults (Kaiming uniform for conv/linear) | Standard and appropriate for ReLU activations. |
| Regularization | None | No dropout, no weight decay, no augmentation — the dataset is large enough relative to the model. |

### Train / test split

No validation set is used during training. The split is fixed at generation time:

| Split | Images | Per class |
|---|---:|---:|
| Train | 2 400 | 800 |
| Test | 450 | 150 |

The model is trained on the 2 400-image training set and evaluated on the 450-image test set after every epoch.

### Environment

`python ≥ 3.10` · `numpy ≥ 1.24` · `matplotlib ≥ 3.7` · `Pillow ≥ 10.0` · `torch ≥ 2.0` · `torchvision ≥ 0.15` · `scikit-learn ≥ 1.3` (confusion matrix only)

---

## Dashboard

### Training scorecard

<table>
<tr>
  <th align="left">Checkpoint</th>
  <th>Train acc</th>
  <th>Test acc</th>
  <th>Test loss</th>
</tr>
<tr>
  <td><b>Epoch 1</b></td>
  <td align="center"><img src="https://img.shields.io/badge/42.5%25-7A7A7A?style=flat-square" alt="42.5%"></td>
  <td align="center"><img src="https://img.shields.io/badge/60.9%25-7A7A7A?style=flat-square" alt="60.9%"></td>
  <td align="center"><img src="https://img.shields.io/badge/0.877-7A7A7A?style=flat-square" alt="0.877"></td>
</tr>
<tr>
  <td><b>Epoch 5</b></td>
  <td align="center"><img src="https://img.shields.io/badge/96.5%25-7A7A7A?style=flat-square" alt="96.5%"></td>
  <td align="center"><img src="https://img.shields.io/badge/97.3%25-7A7A7A?style=flat-square" alt="97.3%"></td>
  <td align="center"><img src="https://img.shields.io/badge/0.097-7A7A7A?style=flat-square" alt="0.097"></td>
</tr>
<tr>
  <td><b>Epoch 10</b></td>
  <td align="center"><img src="https://img.shields.io/badge/99.9%25-7A7A7A?style=flat-square" alt="99.9%"></td>
  <td align="center"><img src="https://img.shields.io/badge/99.8%25-7A7A7A?style=flat-square" alt="99.8%"></td>
  <td align="center"><img src="https://img.shields.io/badge/0.0135-7A7A7A?style=flat-square" alt="0.0135"></td>
</tr>
<tr>
  <td><b>Epoch 15</b> <sub>(final)</sub></td>
  <td align="center"><img src="https://img.shields.io/badge/100%25-3B6EA8?style=flat-square" alt="100%"></td>
  <td align="center"><img src="https://img.shields.io/badge/99.6%25-3B6EA8?style=flat-square" alt="99.6%"></td>
  <td align="center"><img src="https://img.shields.io/badge/0.0060-3B6EA8?style=flat-square" alt="0.0060"></td>
</tr>
</table>

<sub>Values from `results/metrics.json` · blue = final epoch · gray = intermediate checkpoints · test errors: 2 wrong out of 450</sub>

### 1. The dataset — synthetic shapes

![samples](assets/01_samples.png)

Three classes, 800 training and 150 test images per class. Each image is 64×64 RGB:

- **Background**: a dark random base color with mild Gaussian noise added pixel-wise (palette-friendly: low-saturation purples / greens / browns).
- **Shape**: rendered with PIL at a random position and size, in a near-white random foreground color, with a small rotation drawn from ±22.5°.
- **No augmentation** at training time — the dataset *is* the variability.

Every byte is generated from the seed in `generate_data.py`. No external download.

### 2. Training dynamics

![training curves](assets/02_curves.png)

Both loss curves drop monotonically and converge to near zero. Crucially, **the test loss tracks the train loss closely** — there's no overfitting bulge. That's the signature of "the model has enough capacity to fit, the dataset has enough breadth to generalize, and the training run is long enough to converge." Three things going right at once.

### 3. Confusion matrix on the test set

![confusion](assets/03_confusion.png)

The diagonal is essentially full. With 150 examples per class, getting 1–2 wrong rounds to 99% per-class precision and recall.

### 4. Misclassified samples — what the model still gets wrong

![misclassified](assets/04_misclassified.png)

Out of 450 test images, **only 2** are misclassified — and both are visually marginal cases that a tired human might also miss. This is the kind of error log you want to see at the end of training: not "everything is wrong in the same way" but "a handful of edge cases the model hasn't fully internalized."

---

## Validation methodology

### Metrics computed

| Metric | Definition | Computed how |
|---|---|---|
| **Train accuracy** | $\frac{\text{correct predictions}}{\text{total training samples}}$ per epoch | Accumulated over mini-batches during the forward pass each epoch |
| **Test accuracy** | $\frac{\text{correct predictions}}{450}$ per epoch | Full test-set forward pass after each epoch with `model.eval()` and `torch.no_grad()` |
| **Cross-entropy loss** | $-\frac{1}{n}\sum_{i}\log p_{y_i}$ where $p_{y_i}$ is the softmax probability of the true class | `F.cross_entropy` on raw logits; averaged over samples in each batch |
| **Confusion matrix** | 3×3 count matrix, row = true class, column = predicted class | `sklearn.metrics.confusion_matrix` on the full test set at the end of training; normalized row-wise for display |

Per-class precision, recall, and F1 are **not written to `results/metrics.json`** — they are visible in the confusion matrix figure (`assets/03_confusion.png`) but not stored numerically. At 99.6% overall accuracy with 2 misclassifications out of 450, per-class precision and recall are all ≥ 98.7%.

### Reading the training curves

The loss and accuracy curves (`assets/02_curves.png`) show two traces each — train (blue) and test (red):

- **Healthy convergence**: both curves descend monotonically and converge to near-zero loss / near-1.0 accuracy. This is what this run shows.
- **Overfitting** would appear as test loss rising while train loss continues to fall. The train–test gap here stays under 0.5 percentage points throughout — negligible.
- **Underfitting** would appear as both curves plateauing far above zero. Not observed here.
- **Convergence point**: test accuracy first crosses 95% at epoch 5 (97.3%), reaches ~99% by epoch 7, and stabilizes at 99.6% by the final epoch.

### Full results

All numbers are taken exactly from `results/metrics.json` as written by `python train.py`.

| Epoch | Train loss | Test loss | Train acc | Test acc |
|---:|---:|---:|---:|---:|
| 1 | 1.0603 | 0.8773 | 42.5% | 60.9% |
| 2 | 0.6878 | 0.5432 | 62.9% | 70.7% |
| 3 | 0.5087 | 0.4162 | 70.9% | 82.9% |
| 4 | 0.3369 | 0.1929 | 86.8% | 94.9% |
| 5 | 0.1245 | 0.0971 | 96.5% | 97.3% |
| 6 | 0.0452 | 0.0453 | 98.7% | 98.9% |
| 7 | 0.0394 | 0.0226 | 98.7% | 99.6% |
| 8 | 0.0123 | 0.0206 | 99.8% | 99.3% |
| 9 | 0.0081 | 0.0139 | 99.9% | 99.8% |
| 10 | 0.0074 | 0.0135 | 99.9% | 99.8% |
| 11 | 0.0044 | 0.0125 | 100.0% | 99.3% |
| 12 | 0.0053 | 0.0079 | 99.8% | 99.8% |
| 13 | 0.0024 | 0.0081 | 100.0% | 99.8% |
| 14 | 0.0017 | 0.0060 | 100.0% | 99.8% |
| **15** | **0.0012** | **0.0060** | **100.0%** | **99.6%** |

<sub>Exact values from `results/metrics.json`. Final: train accuracy 100.0%, test accuracy 99.6% (2 wrong out of 450).</sub>

### Reproducibility

- **Determinism.** `generate_data.py` uses `numpy.random.default_rng(seed=42)`. `train.py` calls `torch.manual_seed(42)` before model construction and data loading. On CPU, these seeds are sufficient to reproduce the numbers above exactly.
- **GPU non-determinism caveat.** If run on a GPU, CUDA operations (particularly `atomicAdd` in backward passes) introduce non-deterministic rounding; results may differ by fractions of a percent. The numbers above were produced on CPU.

---

## What's actually happening

### Architecture — 3 conv blocks → flatten → 2-layer MLP head

```
Input (3, 64, 64)
  │
  ├─ Conv 3→16, ReLU, Conv 16→16, ReLU, MaxPool 2×2  →  (16, 32, 32)
  ├─ Conv 16→32, ReLU,           MaxPool 2×2          →  (32, 16, 16)
  ├─ Conv 32→64, ReLU,           MaxPool 2×2          →  (64,  8,  8)
  │
  └─ Flatten (4096) → Linear → ReLU → Linear → 3 logits
```

About 540K parameters total — small by 2025 standards, but enough for this 3-class task. The receptive field after the third conv block covers ~28×28 of the input, which is much more than the typical shape size.

### Training recipe

| Knob | Value | Why |
|---|---|---|
| Optimizer | Adam, lr = 5e-4 | Stable for small networks; no warmup needed at this scale |
| Batch size | 32 | Smaller batches = noisier gradient = mild regularization |
| Loss | Cross-entropy | Standard multiclass classification |
| Epochs | 15 | Convergence is reached by epoch ≈ 8; the tail epochs verify stability |
| Regularization | None (no dropout, no augmentation) | The dataset is large enough relative to the model that explicit regularization isn't required |

### What this is not

This is intentionally not a CIFAR-10 / ImageNet-scale demo. It's a *minimal vertical slice* of "build a CNN from scratch in PyTorch": dataset generation → DataLoader → model definition → training loop → evaluation → plotting. The point is to make every step visible and modifiable, not to push a benchmark. A natural extension is to substitute in a real benchmark dataset and observe what changes (BatchNorm becomes essential, augmentation starts mattering, training takes longer).

### Why we removed BatchNorm

An earlier version of this CNN used BatchNorm after every conv. With only ~75 training batches per epoch and a batch size of 32, the BN running statistics never stabilized — the network sat at random accuracy for the entire run. Removing BN fixed it. Lesson: **BatchNorm is not a free win on small datasets**; on tiny ones, GroupNorm or no normalization is often more reliable.

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py    # render 2400 train + 450 test images (deterministic)
python train.py            # 15 epochs of training + dashboard figures
```

Total wall-time: ~30–60 seconds on a modern CPU (no GPU needed).

### Tweak the difficulty

`DataConfig` in [`generate_data.py`](generate_data.py):

```python
DataConfig(
    image_size=64,
    n_per_class_train=800,    # training set size per class
    n_per_class_test=150,
    bg_noise_std=6.0,         # raise to make the task harder
    shape_size_min=0.30,      # smaller shapes = harder
    shape_size_max=0.50,
    seed=42,
)
```

To make the task genuinely difficult, raise `bg_noise_std` to ~25, drop `shape_size_min` to 0.10, and bump rotation range to ±π in `generate_data.py`. The same architecture will then need BatchNorm, dropout, and data augmentation to recover.

---

## Project layout

```
05-cnn-image-classification/
├── README.md              ← this dashboard
├── requirements.txt
├── generate_data.py       ← synthetic shape renderer (PIL + numpy)
├── train.py               ← model + training loop + dashboard figures
├── assets/                ← rendered dashboard figures (4 PNGs)
└── results/metrics.json
```

---

## Notes on methodology & limitations

Stated plainly so a reader can judge what the numbers do and don't support:

- **Synthetic shapes are much easier than natural images.** Circles, squares, and triangles on a dark background have stable, high-contrast boundaries with no texture, lighting variation, or intra-class appearance spread. A network that reaches 99.6% here has not demonstrated anything close to general vision ability — it has learned to detect edges and enclosed regions in a tightly controlled setting. The same architecture scores ~65–70% on CIFAR-10 without significant modifications.
- **Near-ceiling accuracy means the task is likely too easy, not that the model is unusually good.** At 99.6% with only 2 errors, the benchmark doesn't discriminate between this model and a wide range of architectures. Harder variants (smaller shapes, heavier noise, overlapping foreground/background color distributions, full rotation) would provide more signal about what the architecture can actually do.
- **Single fixed train/test split, no cross-validation.** One 2400/450 split is sufficient to demonstrate the training pipeline, but a single split can be lucky or unlucky. Reported accuracy could vary by ±0.5% across seeds. A rigorous benchmark would report mean ± std over multiple seeds or k-fold CV.
- **No data augmentation.** Variability is injected at generation time, so the training images are already diverse — but the distribution of test images is the same as training, generated from the same process. On real tasks, test distribution shift is common and augmentation is the standard mitigation. Here it would have little effect.
- **Architecture is small and untuned.** The ~540 K-parameter CNN with no BatchNorm or Dropout was chosen for pedagogical clarity and to illustrate the BatchNorm pitfall on small datasets. It is not a competitive architecture by any other standard; the results say more about the task than about the model.

---

## What I learned

- **A working model on a manageable problem teaches more than a half-working model on a hard one.** I went around the loop several times trying to make the shape task harder (random colors, full rotation, heavy background noise) and the model collapsed each time. Pulling back to a well-posed task and *getting clean curves* was the better learning experience — and a more honest portfolio piece.
- **BatchNorm is not "always on, always helps."** On tiny datasets where each epoch only sees ~75 batches, BN's running statistics drift between train and eval modes and the network never escapes its initial random state. The fix was simple — drop BN — and the lesson is that *normalization layers are domain-dependent*.
- **The misclassified-sample gallery is the real test report.** The bar chart says 99.6%, but seeing the two specific images the model got wrong tells you whether the remaining errors are "the model is confused about a class" or "this image was always going to be hard." The latter is fine; the former is a real bug.
- **PyTorch's training loop is short — the *operations around it* are most of the code.** Out of 200 lines in `train.py`, the actual `for batch in loader: forward; backward; step` block is six lines. Dataset prep, evaluation, plotting, and metrics-saving are everything else. That ratio doesn't change much for production code.

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Data is fully synthetic and self-generated.</sub>
</div>
