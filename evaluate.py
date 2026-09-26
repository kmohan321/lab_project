"""
evaluate.py
Evaluation module for BloodANN on BloodMNIST test set.
Computes test metrics, classification report, confusion matrix, and visual predictions.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders, CLASS_NAMES
from model import BloodANN
from visualize import plot_confusion_matrix, plot_sample_predictions


def evaluate_test_set(
    checkpoint_path="./checkpoints/best_model.pth",
    data_dir="./data",
    batch_size=64,
    n_display_samples=16
):
    """
    Evaluates the trained BloodANN checkpoint on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    # Load dataloaders
    _, _, test_loader, class_names = get_dataloaders(data_dir=data_dir, batch_size=batch_size, multiplier=1)

    # Initialize model and load weights
    model = BloodANN(in_features=3 * 28 * 28, num_classes=len(class_names)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[*] Successfully loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint and 'val_acc' in checkpoint:
        print(f"[*] Checkpoint details -> Epoch: {checkpoint['epoch']}, Best Val Acc: {checkpoint['val_acc'] * 100:.2f}%")

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_confs = []

    # Store first batch for sample visualization
    sample_images = []
    sample_targets = []
    sample_preds = []
    sample_confs = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

            if len(sample_images) < n_display_samples:
                needed = n_display_samples - len(sample_images)
                sample_images.extend(images[:needed].cpu())
                sample_targets.extend(targets[:needed].cpu().numpy())
                sample_preds.extend(preds[:needed].cpu().numpy())
                sample_confs.extend(confs[:needed].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confs = np.array(all_confs)

    test_loss = total_loss / len(all_targets)
    test_acc = np.mean(all_preds == all_targets)

    print("\n" + "=" * 65)
    print(f"  BLOODMNIST TEST SET EVALUATION RESULTS (N = {len(all_targets):,})")
    print("=" * 65)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print("=" * 65)

    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names=class_names, save_path="confusion_matrix.png")

    # Sample Predictions Plot
    plot_sample_predictions(
        images=sample_images,
        true_labels=sample_targets,
        pred_labels=sample_preds,
        confidences=sample_confs,
        class_names=class_names,
        save_path="sample_predictions.png",
        n_samples=n_display_samples
    )

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'report': report,
        'confusion_matrix': cm.tolist()
    }


if __name__ == "__main__":
    evaluate_test_set()
