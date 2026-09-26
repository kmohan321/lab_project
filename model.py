"""
model.py
Deep Artificial Neural Network (ANN / Multi-Layer Perceptron) for BloodMNIST classification.
"""

import torch
import torch.nn as nn


class BloodANN(nn.Module):
    """
    Deep Artificial Neural Network (MLP) architecture designed for BloodMNIST image classification.
    
    Inputs: Flattened RGB images (3 x 28 x 28 = 2352 features).
    Hidden Layers: 4 Dense blocks with Batch Normalization, LeakyReLU activation, and Dropout.
    Output: 8 logits corresponding to the blood cell categories.
    """
    def __init__(self, in_features=3 * 28 * 28, num_classes=8, dropout_rate=0.25):
        super(BloodANN, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        self.network = nn.Sequential(
            # Flatten 3x28x28 -> 2352
            nn.Flatten(),

            # Block 1: 2352 -> 1024
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout_rate + 0.05),

            # Block 2: 1024 -> 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout_rate),

            # Block 3: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout_rate - 0.05),

            # Block 4: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout_rate - 0.10),

            # Output Head: 128 -> 8 classes
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for LeakyReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        """Returns total and trainable parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    model = BloodANN()
    total, trainable = model.count_parameters()
    print(f"BloodANN initialized successfully.")
    print(f"Total parameters: {total:,} | Trainable: {trainable:,}")
    dummy_input = torch.randn(4, 3, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: [4, 8])")
