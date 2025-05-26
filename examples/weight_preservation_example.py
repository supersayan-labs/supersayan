"""
Example demonstrating how to replace pre-trained PyTorch layers with Supersayan layers
while preserving the trained weights.
"""

import torch
import torch.nn as nn
from supersayan.nn.layers import (
    Linear,
    Conv2d,
    from_pytorch_linear,
    from_pytorch_conv2d,
)


def manual_conversion_example():
    """Example of manually converting layers with weight preservation."""

    # Create a pre-trained PyTorch model
    pytorch_model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

    # Simulate training (in reality, you'd load pre-trained weights)
    # pytorch_model.load_state_dict(torch.load('pretrained_model.pth'))

    # Manual conversion - passing weights directly
    supersayan_linear1 = Linear(
        in_features=784,
        out_features=256,
        weight=pytorch_model[0].weight.data,
        bias=True,
        bias_values=pytorch_model[0].bias.data,
    )

    supersayan_linear2 = Linear(
        in_features=256,
        out_features=10,
        weight=pytorch_model[2].weight.data,
        bias=True,
        bias_values=pytorch_model[2].bias.data,
    )

    print("Manual conversion completed!")
    print(f"PyTorch weight shape: {pytorch_model[0].weight.shape}")
    print(f"Supersayan weight shape: {supersayan_linear1.weight.shape}")
    print(
        f"Weights are equal: {torch.allclose(pytorch_model[0].weight, supersayan_linear1.weight)}"
    )


def utility_function_example():
    """Example using the utility functions for conversion."""

    # Create pre-trained PyTorch layers
    pytorch_linear = nn.Linear(128, 64)
    pytorch_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

    # Convert using utility functions
    supersayan_linear = from_pytorch_linear(pytorch_linear)
    supersayan_conv = from_pytorch_conv2d(pytorch_conv)

    print("\nUtility function conversion completed!")
    print(
        f"Linear - Weights preserved: {torch.allclose(pytorch_linear.weight, supersayan_linear.weight)}"
    )
    print(
        f"Conv2d - Weights preserved: {torch.allclose(pytorch_conv.weight, supersayan_conv.weight)}"
    )


def full_model_conversion_example():
    """Example of converting an entire model."""

    class PyTorchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)

    class SupersayanCNN(nn.Module):
        def __init__(self, pytorch_model):
            super().__init__()
            # Convert each layer, preserving weights
            self.conv1 = Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                weight=pytorch_model.conv1.weight.data,
                bias_values=pytorch_model.conv1.bias.data,
            )
            self.conv2 = Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                weight=pytorch_model.conv2.weight.data,
                bias_values=pytorch_model.conv2.bias.data,
            )
            self.fc1 = from_pytorch_linear(pytorch_model.fc1)
            self.fc2 = from_pytorch_linear(pytorch_model.fc2)

    # Create and convert model
    pytorch_model = PyTorchCNN()
    # In practice: pytorch_model.load_state_dict(torch.load('trained_model.pth'))

    supersayan_model = SupersayanCNN(pytorch_model)

    print("\nFull model conversion completed!")
    print("All layers converted with weights preserved!")


if __name__ == "__main__":
    print("=== Supersayan Weight Preservation Examples ===\n")

    manual_conversion_example()
    utility_function_example()
    full_model_conversion_example()
