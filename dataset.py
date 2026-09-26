"""
dataset.py
Data loading, domain-specific biological augmentations, and sample expansion multiplier
for the BloodMNIST dataset (MedMNIST v2).
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import BloodMNIST

CLASS_NAMES = [
    'Basophil',
    'Eosinophil',
    'Erythroblast',
    'Immature Granulocyte',
    'Lymphocyte',
    'Monocyte',
    'Neutrophil',
    'Platelet'
]

# Normalization constants (MedMNIST standard: map [0, 1] to [-1, 1])
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


def get_transforms():
    """
    Returns data transformations for training (biological augmentations)
    and evaluation/testing (clean normalization).
    """
    train_augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return train_augment_transform, eval_transform


class AugmentedExpandedDataset(Dataset):
    """
    Multiplies the effective dataset size by generating diverse stochastic augmentations
    for each sample replica during training.
    
    If multiplier = 3:
      - Replica 0 (indices 0 .. N-1): Base image with standard normalization.
      - Replica 1 & 2 (indices N .. 3N-1): Stochastic biologically augmented variants.
    """
    def __init__(self, base_dataset, base_transform, augment_transform, multiplier=3):
        self.base_dataset = base_dataset
        self.base_transform = base_transform
        self.augment_transform = augment_transform
        self.multiplier = max(1, int(multiplier))
        self.base_len = len(base_dataset)

    def __len__(self):
        return self.base_len * self.multiplier

    def __getitem__(self, idx):
        base_idx = idx % self.base_len
        replica_idx = idx // self.base_len

        img, target = self.base_dataset[base_idx]

        # Ensure image is in PIL format before transform
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # Apply appropriate transformation
        if replica_idx == 0:
            transformed_img = self.base_transform(img)
        else:
            transformed_img = self.augment_transform(img)

        if isinstance(target, np.ndarray):
            label = int(target.squeeze())
        elif isinstance(target, torch.Tensor):
            label = int(target.squeeze().item())
        else:
            label = int(target)

        return transformed_img, label


def get_data_root(data_dir=None):
    """
    Finds or creates a valid directory containing bloodmnist.npz.
    """
    if data_dir is not None and os.path.exists(os.path.join(data_dir, "bloodmnist.npz")):
        return data_dir
    default_home = os.path.expanduser("~/.medmnist")
    if os.path.exists(os.path.join(default_home, "bloodmnist.npz")):
        return default_home
    target_dir = data_dir if data_dir is not None else default_home
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def get_dataloaders(data_dir=None, batch_size=64, multiplier=3, num_workers=0):
    """
    Initializes and returns train, validation, and test DataLoaders for BloodMNIST.
    """
    root_dir = get_data_root(data_dir)
    train_augment_tf, eval_tf = get_transforms()
    should_download = not os.path.exists(os.path.join(root_dir, "bloodmnist.npz"))

    # Load raw MedMNIST datasets (as PIL images without initial transforms)
    raw_train_ds = BloodMNIST(split='train', root=root_dir, download=should_download, as_rgb=True)
    raw_val_ds = BloodMNIST(split='val', root=root_dir, download=should_download, as_rgb=True)
    raw_test_ds = BloodMNIST(split='test', root=root_dir, download=should_download, as_rgb=True)

    # Wrap training dataset with sample expansion multiplier
    train_dataset = AugmentedExpandedDataset(
        base_dataset=raw_train_ds,
        base_transform=eval_tf,
        augment_transform=train_augment_tf,
        multiplier=multiplier
    )

    # Validation and Test datasets (multiplier=1, only evaluation transforms)
    val_dataset = AugmentedExpandedDataset(
        base_dataset=raw_val_ds,
        base_transform=eval_tf,
        augment_transform=eval_tf,
        multiplier=1
    )

    test_dataset = AugmentedExpandedDataset(
        base_dataset=raw_test_ds,
        base_transform=eval_tf,
        augment_transform=eval_tf,
        multiplier=1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, CLASS_NAMES
