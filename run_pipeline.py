"""
run_pipeline.py
Master execution script for the BloodMNIST ANN classification pipeline.
Executes dataset preparation, biological augmentation visualization,
training with dynamic sample expansion, and rigorous test evaluation.
"""

import argparse
import os
import sys
import torch
import medmnist
from medmnist import BloodMNIST

from dataset import get_transforms, get_data_root, CLASS_NAMES
from visualize import plot_augmentation_comparison
from train import train_model
from evaluate import evaluate_test_set


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ANN for BloodMNIST Classification with Augmentation Expansion")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where BloodMNIST dataset is stored")
    parser.add_argument("--multiplier", type=int, default=3, help="Dataset sample expansion multiplier (e.g. 3 = 35,877 samples/epoch)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 weight decay penalty")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and evaluate existing checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  BLOODMNIST CLASSIFICATION PIPELINE (ARTIFICIAL NEURAL NETWORK)")
    print("=" * 70)
    print(f"  Configuration:")
    print(f"  - Dataset Multiplier: {args.multiplier}x (Effective Train Samples: ~{args.multiplier * 11959:,})")
    print(f"  - Epochs:             {args.epochs}")
    print(f"  - Batch Size:         {args.batch_size}")
    print(f"  - Learning Rate:      {args.lr}")
    print(f"  - Weight Decay:       {args.weight_decay}")
    print(f"  - CUDA Available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU Device:         {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Step 1: Download / Verify Dataset and Plot Augmentations
    print("\n[Step 1/3] Preparing dataset and generating biological augmentation preview...")
    root_dir = get_data_root(args.data_dir)
    should_download = not os.path.exists(os.path.join(root_dir, "bloodmnist.npz"))
    raw_train_ds = BloodMNIST(split='train', root=root_dir, download=should_download, as_rgb=True)
    train_augment_tf, _ = get_transforms()
    plot_augmentation_comparison(raw_train_ds, train_augment_tf, class_names=CLASS_NAMES, save_path="augmented_samples.png")

    best_checkpoint = os.path.join(args.checkpoint_dir, "best_model.pth")

    # Step 2: Training
    if not args.skip_train:
        print("\n[Step 2/3] Training BloodANN model...")
        best_checkpoint, history = train_model(
            data_dir=args.data_dir,
            multiplier=args.multiplier,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir
        )
    else:
        print(f"\n[Step 2/3] Skipping training as requested. Using checkpoint: {best_checkpoint}")

    # Step 3: Test Evaluation
    print("\n[Step 3/3] Evaluating on independent BloodMNIST test set...")
    metrics = evaluate_test_set(
        checkpoint_path=best_checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE!")
    print("=" * 70)
    print(f"  Artifacts generated:")
    print(f"  1. augmented_samples.png   - Biological augmentation visual comparison")
    print(f"  2. learning_curves.png       - Training & validation loss and accuracy curves")
    print(f"  3. confusion_matrix.png      - Test set confusion matrix with rates & counts")
    print(f"  4. sample_predictions.png   - Visual test predictions with ground truth")
    print(f"  5. history.json              - Full training log")
    print(f"  6. {best_checkpoint}   - Best model weights checkpoint")
    print("=" * 70)


if __name__ == "__main__":
    main()
