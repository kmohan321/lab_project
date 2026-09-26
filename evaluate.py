"""
evaluate.py
Comprehensive evaluation module for BloodANN on the BloodMNIST test set.
Computes exhaustive metrics (Precision, Recall, Specificity, F1, Balanced Accuracy,
ROC-AUC, PR-AUC, Cohen's Kappa, MCC, Top-2 Accuracy), exports structured data to
JSON and CSV files, and generates visual diagnostics.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    top_k_accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

from dataset import get_dataloaders, CLASS_NAMES
from model import BloodANN
from visualize import (
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_per_class_metrics_bar
)


def evaluate_test_set(
    checkpoint_path="./checkpoints/best_model.pth",
    data_dir="./data",
    batch_size=64,
    n_display_samples=16,
    output_json="test_metrics.json",
    output_csv="test_metrics.csv",
    output_txt="test_metrics_summary.txt"
):
    """
    Evaluates the trained BloodANN checkpoint on the test set and outputs full metric files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    # Load dataloaders
    _, _, test_loader, class_names = get_dataloaders(data_dir=data_dir, batch_size=batch_size, multiplier=1)
    num_classes = len(class_names)

    # Initialize model and load weights
    model = BloodANN(in_features=3 * 28 * 28, num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[*] Successfully loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint and 'val_acc' in checkpoint:
        print(f"[*] Checkpoint info -> Epoch: {checkpoint['epoch']}, Best Val Acc: {checkpoint['val_acc'] * 100:.2f}%")

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_confs = []
    all_probs = []

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
            all_probs.extend(probs.cpu().numpy())

            if len(sample_images) < n_display_samples:
                needed = n_display_samples - len(sample_images)
                sample_images.extend(images[:needed].cpu())
                sample_targets.extend(targets[:needed].cpu().numpy())
                sample_preds.extend(preds[:needed].cpu().numpy())
                sample_confs.extend(confs[:needed].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confs = np.array(all_confs)
    all_probs = np.array(all_probs)
    n_samples = len(all_targets)

    # --- 1. Global Metrics ---
    test_loss = total_loss / n_samples
    test_acc = float(np.mean(all_preds == all_targets))
    top2_acc = float(top_k_accuracy_score(all_targets, all_probs, k=2))
    kappa = float(cohen_kappa_score(all_targets, all_preds))
    mcc = float(matthews_corrcoef(all_targets, all_preds))

    # Binarized targets for ROC/PR calculations
    y_true_bin = label_binarize(all_targets, classes=list(range(num_classes)))
    macro_roc_auc = float(roc_auc_score(y_true_bin, all_probs, average="macro"))
    weighted_roc_auc = float(roc_auc_score(y_true_bin, all_probs, average="weighted"))
    macro_pr_auc = float(average_precision_score(y_true_bin, all_probs, average="macro"))
    weighted_pr_auc = float(average_precision_score(y_true_bin, all_probs, average="weighted"))

    # Macro & Weighted precision, recall, f1
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # --- 2. Exhaustive Per-Class Metrics ---
    per_class_data = []
    for c in range(num_classes):
        c_name = class_names[c]
        support = int(np.sum(all_targets == c))
        pred_count = int(np.sum(all_preds == c))

        tp = int(np.sum((all_targets == c) & (all_preds == c)))
        fp = int(np.sum((all_targets != c) & (all_preds == c)))
        fn = int(np.sum((all_targets == c) & (all_preds != c)))
        tn = int(np.sum((all_targets != c) & (all_preds != c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = (recall + specificity) / 2.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        try:
            cls_roc_auc = float(roc_auc_score(y_true_bin[:, c], all_probs[:, c]))
        except Exception:
            cls_roc_auc = 0.0

        try:
            cls_pr_auc = float(average_precision_score(y_true_bin[:, c], all_probs[:, c]))
        except Exception:
            cls_pr_auc = 0.0

        per_class_data.append({
            "Class_ID": c,
            "Class": c_name,
            "Support": support,
            "Predicted_Count": pred_count,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Specificity": round(specificity, 4),
            "Balanced_Accuracy": round(balanced_acc, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(cls_roc_auc, 4),
            "PR-AUC": round(cls_pr_auc, 4)
        })

    # Save to CSV
    df_per_class = pd.DataFrame(per_class_data)
    df_per_class.to_csv(output_csv, index=False)
    print(f"[+] Saved per-class metrics CSV to: {output_csv}")

    # Assemble complete dictionary
    full_metrics = {
        "dataset": "BloodMNIST (MedMNIST v2)",
        "model": "BloodANN (Deep Fully Connected MLP)",
        "test_samples": n_samples,
        "global_metrics": {
            "test_loss": round(test_loss, 4),
            "accuracy": round(test_acc, 4),
            "top2_accuracy": round(top2_acc, 4),
            "cohen_kappa": round(kappa, 4),
            "matthews_corrcoef": round(mcc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "macro_roc_auc": round(macro_roc_auc, 4),
            "macro_pr_auc": round(macro_pr_auc, 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "weighted_roc_auc": round(weighted_roc_auc, 4),
            "weighted_pr_auc": round(weighted_pr_auc, 4),
        },
        "per_class_metrics": per_class_data,
        "confusion_matrix": cm.tolist()
    }

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(full_metrics, f, indent=4)
    print(f"[+] Saved complete evaluation JSON to: {output_json}")

    # Standard classification report string
    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

    # Save formatted text summary
    summary_text = f"""======================================================================
  BLOODMNIST TEST SET COMPREHENSIVE EVALUATION REPORT
======================================================================
Total Test Samples: {n_samples:,}
Cross-Entropy Loss: {test_loss:.4f}
Top-1 Accuracy:     {test_acc * 100:.2f}%
Top-2 Accuracy:     {top2_acc * 100:.2f}%
Cohen's Kappa:      {kappa:.4f}
Matthews Corr (MCC):{mcc:.4f}

Macro ROC-AUC:      {macro_roc_auc:.4f}
Weighted ROC-AUC:   {weighted_roc_auc:.4f}
Macro PR-AUC (mAP): {macro_pr_auc:.4f}
Weighted PR-AUC:    {weighted_pr_auc:.4f}

======================================================================
PER-CLASS DETAILED BREAKDOWN (PRECISION, RECALL, SPECIFICITY, F1, AUC)
======================================================================
{df_per_class[['Class', 'Support', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC', 'PR-AUC']].to_string(index=False)}

======================================================================
STANDARD CLASSIFICATION REPORT:
======================================================================
{report_str}
======================================================================
"""
    with open(output_txt, 'w') as f:
        f.write(summary_text)
    print(f"[+] Saved text summary to: {output_txt}")

    # Print to console
    print("\n" + summary_text)

    # --- 3. Generate Evaluation Diagnostic Plots ---
    plot_confusion_matrix(cm, class_names=class_names, save_path="confusion_matrix.png")
    plot_roc_curves(all_targets, all_probs, class_names=class_names, save_path="roc_curves.png")
    plot_precision_recall_curves(all_targets, all_probs, class_names=class_names, save_path="precision_recall_curves.png")
    plot_per_class_metrics_bar(per_class_data, save_path="per_class_metrics_bar.png")
    plot_sample_predictions(
        images=sample_images,
        true_labels=sample_targets,
        pred_labels=sample_preds,
        confidences=sample_confs,
        class_names=class_names,
        save_path="sample_predictions.png",
        n_samples=n_display_samples
    )

    return full_metrics


if __name__ == "__main__":
    evaluate_test_set()
