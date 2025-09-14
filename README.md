
# MobileNetV2 on Fashion-MNIST and CIFAR-10 — Colab Training & Evaluation

This notebook trains **MobileNetV2** **from scratch** on two datasets: **Fashion-MNIST** and **CIFAR-10**, then evaluates and compares results.

---

## 1) Quick Start (Colab)

1. Open Google Colab → Runtime → Change runtime type → GPU (recommended) I used A100.
2. Upload the provided notebook or paste the code cells in order.
3. Run all cells. Training is **50 epochs per dataset**, no early stopping.
4. Outputs include:
   - Saved models: `fashion_mobilenetv2.h5`, `cifar10_mobilenetv2.h5`
   - Accuracy, Precision, Recall, F1-score reports
   - Confusion matrices (heatmaps)
   - Training history plots (accuracy & loss curves)

---

## 2) What’s Inside

- **Datasets**:
  - *Fashion-MNIST*: 60k train / 10k test grayscale clothing images (28×28). Resized to 32×32×3.
  - *CIFAR-10*: 50k train / 10k test color images (32×32×3).
- **Splits**:
  - Fashion-MNIST: 48k train, 12k validation, 10k test.
  - CIFAR-10: 45k train, 5k validation, 10k test.
- **Augmentation**: random horizontal flip, ±10% rotation, ±10% zoom.
- **Preprocessing**: rescale pixels to [0,1] then to [-1,1] for MobileNetV2 compatibility.
- **Model**: `tf.keras.applications.MobileNetV2` with `weights=None`, `input_shape=(32,32,3)`, `classes=10`.
- **Training**: Adam optimizer, learning rate=0.001, batch size=64, 50 epochs.
- **Evaluation**:
  - Accuracy, precision, recall, F1-score (per class + averages)
  - Confusion matrix plots
  - Training/validation curves

---

## 3) Steps

1. **Load & Preprocess Data**: Resize Fashion-MNIST, duplicate channels, normalize to [0,1].
2. **Split Data**: training, validation, and test sets created.
3. **Build Model**: MobileNetV2 + preprocessing layers.
4. **Train**: 50 epochs, batch size=64, record history.
5. **Save Models**: export `.h5` files.
6. **Evaluate**: test metrics, confusion matrix, classification report.
7. **Visualize**: training curves and confusion matrices.

---

## 4) Outputs

- `fashion_mobilenetv2.h5`
- `cifar10_mobilenetv2.h5`
- Accuracy & loss plots for both datasets
- Classification reports & confusion matrices

---

## 5) Expected Results (ballpark)

- **Fashion-MNIST**:
  - Test accuracy: ~90–92%
  - F1-score (macro): ~0.90
- **CIFAR-10**:
  - Test accuracy: ~70–80% (from scratch, 50 epochs)
  - F1-score (macro): ~0.70–0.75

---

## 6) Extend / Modify

- Increase epochs for CIFAR-10 (100+) for higher accuracy.
- Add stronger augmentation (color jitter, random crop).
- Try transfer learning with `weights='imagenet'` for better CIFAR-10 results.
- Adjust learning rate scheduler for improved convergence.

---

## 7) License & Dataset Info

- Fashion-MNIST: by Zalando Research, MIT license.
- CIFAR-10: Alex Krizhevsky & Geoffrey Hinton, 2009.
- Code: TensorFlow/Keras (Apache 2.0).

---

## 8) TL;DR

- MobileNetV2 achieves **higher accuracy** on Fashion-MNIST vs CIFAR-10 (dataset complexity difference).
- Training is **fast** (lightweight model, ~2.2M parameters).
- Use transfer learning or deeper nets (ResNet) for higher CIFAR-10 performance.
