# BloodMNIST Image Classification using Deep Artificial Neural Network (ANN)

An end-to-end deep learning pipeline for peripheral blood cell classification on the **BloodMNIST** dataset (from the **MedMNIST v2** benchmark) using an Artificial Neural Network (ANN / Multi-Layer Perceptron) with **domain-specific biological image augmentations** and a **dynamic sample expansion multiplier**.

---

## 🔬 Dataset Overview (BloodMNIST)

The **BloodMNIST** dataset consists of individual normal peripheral blood cells organized into 8 classes:
- **0: Basophil**
- **1: Eosinophil**
- **2: Erythroblast**
- **3: Immature Granulocytes** (myelocytes, metamyelocytes, promyelocytes)
- **4: Lymphocyte**
- **5: Monocyte**
- **6: Neutrophil**
- **7: Platelet**

### Dataset Resolution & Split:
- **Image Dimensions**: $28 \times 28 \times 3$ (RGB)
- **Train Set**: 11,959 samples
- **Validation Set**: 1,712 samples
- **Test Set**: 3,421 samples
- **Total Images**: 17,092 images

---

## 🧪 Sample Expansion Strategy & Biological Augmentation

Because microscopic cell images have distinct physical properties compared to standard natural photos, we apply domain-informed augmentations and an expansion multiplier:

1. **Sample Expansion Multiplier (`AugmentedExpandedDataset`)**:
   - The user noted the training sample count constraint.
   - We implemented an `AugmentedExpandedDataset` wrapper with a configurable multiplier $M$ (e.g. $M = 3$ or $M = 4$).
   - For $M=3$, each epoch dynamically processes **$35,877$ samples**, where replicas receive independent stochastic biological transformations on the fly.

2. **Domain-Specific Biological Augmentations**:
   - **Isotropic Orientation ($0^\circ - 360^\circ$)**: Microscopic blood smears possess no preferred gravity orientation. We apply `RandomRotation(degrees=180)`, `RandomHorizontalFlip(p=0.5)`, and `RandomVerticalFlip(p=0.5)`.
   - **Stain Variation Simulation**: Wright-Giemsa staining intensity and illumination vary across blood smears. We apply `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05)`.
   - **Cell Positioning & Scale**: Cells may be slightly off-center or magnified differently under microscope objectives. We apply `RandomAffine(degrees=15, translate=(0.06, 0.06), scale=(0.95, 1.05))`.

---

## 🧠 Model Architecture (`BloodANN`)

A 4-hidden-layer deep Fully Connected Artificial Neural Network with Batch Normalization, LeakyReLU, and progressive Dropout:

```
Input (3 x 28 x 28 = 2,352 features)
  │
  ▼
[Flatten Layer]
  │
  ▼
Linear(2352 -> 1024) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.30)
  │
  ▼
Linear(1024 -> 512)  -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.25)
  │
  ▼
Linear(512  -> 256)  -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.20)
  │
  ▼
Linear(256  -> 128)  -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.15)
  │
  ▼
Linear(128  -> 8)    -> Logits (8 Classes)
```

---

## 🚀 How to Run

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run training and evaluation with the default $3\times$ sample multiplier on GPU:
```bash
python run_pipeline.py --multiplier 3 --epochs 25 --batch_size 64 --lr 0.001
```

### 3. Individual Execution
- **Preview Augmentations**:
  ```bash
  python -c "from dataset import get_dataloaders, get_transforms, CLASS_NAMES; from visualize import plot_augmentation_comparison; from medmnist import BloodMNIST; ds = BloodMNIST(split='train', root='./data', download=True, as_rgb=True); tf, _ = get_transforms(); plot_augmentation_comparison(ds, tf, CLASS_NAMES)"
  ```
- **Train Only**:
  ```bash
  python train.py
  ```
- **Evaluate Checkpoint**:
  ```bash
  python evaluate.py
  ```

---

## 📊 Generated Artifacts
1. `augmented_samples.png`: Side-by-side comparison of original vs augmented blood cells.
2. `learning_curves.png`: Training and validation loss/accuracy across epochs.
3. `confusion_matrix.png`: Normalized test set confusion matrix with per-class accuracy.
4. `sample_predictions.png`: Visual grid of test predictions with confidence and correctness indicators.
5. `checkpoints/best_model.pth`: PyTorch model weights checkpoint.
6. `history.json`: Epoch-by-epoch training and validation metrics.
