"""
train.py
Training routine for BloodANN on BloodMNIST with sample expansion multiplier.
"""

import os
import json
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from model import BloodANN
from visualize import plot_learning_curves


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Runs one full training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping for stable training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({
            'loss': f"{running_loss / total:.4f}",
            'acc': f"{(correct / total) * 100:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Evaluates the model on the validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_model(
    data_dir="./data",
    multiplier=3,
    epochs=25,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    checkpoint_dir="./checkpoints",
    history_file="history.json"
):
    """
    Main training function orchestrating dataloaders, optimization, and checkpointing.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        multiplier=multiplier
    )

    base_train_len = len(train_loader.dataset.base_dataset)
    expanded_train_len = len(train_loader.dataset)
    print(f"[*] Base training samples: {base_train_len:,}")
    print(f"[*] Multiplier factor: {multiplier}x")
    print(f"[*] Effective training samples per epoch: {expanded_train_len:,}")
    print(f"[*] Validation samples: {len(val_loader.dataset):,} | Test samples: {len(test_loader.dataset):,}")

    # Build Model
    model = BloodANN(in_features=3 * 28 * 28, num_classes=len(class_names), dropout_rate=0.25).to(device)
    total_params, trainable_params = model.count_parameters()
    print(f"[*] Model parameters: Total = {total_params:,} | Trainable = {trainable_params:,}")

    # Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    start_time = time.time()
    print("\n" + "=" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^11} | {'Val Loss':^11} | {'Val Acc':^11} | {'LR':^9}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Check for best validation accuracy
        is_best = val_acc > best_val_acc
        star = ""
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'multiplier': multiplier
            }, best_model_path)
            star = " * (Best)"

        print(f"{epoch:^7d} | {train_loss:^11.4f} | {train_acc * 100:^10.2f}% | {val_loss:^11.4f} | {val_acc * 100:^10.2f}% | {current_lr:^9.2e}{star}")

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"[+] Training completed in {total_time / 60:.2f} minutes.")
    print(f"[+] Best Validation Accuracy: {best_val_acc * 100:.2f}% (Saved to {best_model_path})")

    # Save training history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"[+] Training history logged to: {history_file}")

    # Plot curves
    plot_learning_curves(history, save_path="learning_curves.png")

    return best_model_path, history


if __name__ == "__main__":
    train_model(epochs=25, multiplier=3)
