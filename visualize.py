"""
visualize.py
Visualization utilities for BloodMNIST:
- Original vs biologically augmented samples
- Training and validation loss/accuracy learning curves
- Normalized confusion matrix
- Sample test predictions with class labels and confidence
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image

from dataset import NORM_MEAN, NORM_STD, CLASS_NAMES


def denormalize_image(tensor):
    """
    Converts a normalized PyTorch tensor (C, H, W) back to a displayable NumPy image (H, W, C) in [0, 1].
    """
    img = tensor.cpu().clone().detach().numpy()
    for c in range(3):
        img[c] = img[c] * NORM_STD[c] + NORM_MEAN[c]
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))


def plot_augmentation_comparison(raw_dataset, augment_transform, class_names=CLASS_NAMES, save_path="augmented_samples.png"):
    """
    Plots a grid comparing original blood cell samples with 3 stochastic augmented variants.
    Shows 1 sample per class (8 classes x 4 columns = 32 images).
    """
    fig, axes = plt.subplots(len(class_names), 4, figsize=(10, 2.3 * len(class_names)))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Instantaneously find one sample index for each class from dataset labels array
    class_indices = {}
    if hasattr(raw_dataset, 'labels'):
        labels_arr = np.array(raw_dataset.labels).squeeze()
        for c in range(len(class_names)):
            matches = np.where(labels_arr == c)[0]
            if len(matches) > 0:
                class_indices[c] = int(matches[0])
    else:
        for idx in range(len(raw_dataset)):
            label = int(raw_dataset[idx][1])
            if label not in class_indices:
                class_indices[label] = idx
            if len(class_indices) == len(class_names):
                break

    for row_idx, cls_label in enumerate(sorted(class_indices.keys())):
        ds_idx = class_indices[cls_label]
        raw_img, _ = raw_dataset[ds_idx]
        if not isinstance(raw_img, Image.Image):
            raw_img = Image.fromarray(raw_img)

        cls_name = class_names[cls_label]

        # Col 0: Original image
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_title(f"Original: {cls_name}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis('off')

        # Col 1, 2, 3: Independently augmented variations
        for col_idx in range(1, 4):
            aug_tensor = augment_transform(raw_img)
            aug_display = denormalize_image(aug_tensor)
            axes[row_idx, col_idx].imshow(aug_display)
            axes[row_idx, col_idx].set_title(f"Augmented #{col_idx}", fontsize=9)
            axes[row_idx, col_idx].axis('off')

    plt.suptitle("BloodMNIST: Original vs Biologically Augmented Blood Cell Samples\n"
                 "(Isotropic Rotation, Horizontal/Vertical Flip, Stain Color Jitter, Affine Shift)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved augmentation comparison to: {save_path}")


def plot_learning_curves(history, save_path="learning_curves.png"):
    """
    Plots training and validation loss and accuracy curves over epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    best_loss_epoch = np.argmin(history['val_loss']) + 1
    best_loss = min(history['val_loss'])
    ax1.scatter(best_loss_epoch, best_loss, color='darkred', s=90, zorder=5,
                label=f'Min Val Loss: {best_loss:.4f} (Ep {best_loss_epoch})')
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # Accuracy curve
    ax2.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, [acc * 100 for acc in history['val_acc']], 'g-s', label='Validation Accuracy', linewidth=2, markersize=4)
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc']) * 100
    ax2.scatter(best_acc_epoch, best_acc, color='darkgreen', s=90, zorder=5,
                label=f'Max Val Acc: {best_acc:.2f}% (Ep {best_acc_epoch})')
    ax2.set_title("Classification Accuracy (%) vs. Epochs", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.suptitle("BloodANN Training Dynamics with Sample Expansion Multiplier", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved learning curves to: {save_path}")


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, save_path="confusion_matrix.png"):
    """
    Plots a normalized confusion matrix heatmap with class counts.
    """
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(9, 7.5))
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"

    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Rate'}
    )
    plt.title("BloodMNIST: Confusion Matrix (Test Set)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("True Class", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved confusion matrix to: {save_path}")


def plot_sample_predictions(images, true_labels, pred_labels, confidences, class_names=CLASS_NAMES, save_path="sample_predictions.png", n_samples=16):
    """
    Plots a grid of test samples with predicted vs ground truth labels.
    """
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
    axes = axes.flatten()

    for i in range(n_samples):
        img_np = denormalize_image(images[i])
        true_cls = class_names[true_labels[i]]
        pred_cls = class_names[pred_labels[i]]
        conf = confidences[i] * 100

        axes[i].imshow(img_np)
        is_correct = (true_labels[i] == pred_labels[i])
        color = 'darkgreen' if is_correct else 'darkred'
        status = '[CORRECT]' if is_correct else '[MISMATCH]'

        title = f"{status}\nPred: {pred_cls} ({conf:.1f}%)\nTrue: {true_cls}"
        axes[i].set_title(title, fontsize=9.5, color=color, fontweight='bold')
        axes[i].axis('off')

    for j in range(n_samples, len(axes)):
        axes[j].axis('off')

    plt.suptitle("BloodMNIST: Model Predictions on Test Blood Cells", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved sample predictions to: {save_path}")
