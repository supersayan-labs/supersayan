import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# Create ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Test input
test_x = np.random.rand(1, 3, 224, 224).astype(np.float32)
x = torch.from_numpy(test_x)

# Track shapes through the network
print("=== ResNet18 Layer-by-Layer Shape Analysis ===")
print(f"Input shape: {x.shape}")

# Manual forward pass to track shapes
with torch.no_grad():
    # Initial layers
    x = model.conv1(x)
    print(f"After conv1: {x.shape}")
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    print(f"After maxpool: {x.shape}")
    
    # Layer 1
    identity = x
    out = model.layer1[0].conv1(x)
    print(f"After layer1.0.conv1: {out.shape}")
    out = model.layer1[0].bn1(out)
    out = model.layer1[0].relu(out)
    out = model.layer1[0].conv2(out)
    print(f"After layer1.0.conv2: {out.shape}")
    out = model.layer1[0].bn2(out)
    out += identity
    out = model.layer1[0].relu(out)
    x = out
    print(f"After layer1.0 (block): {x.shape}")
    
    # Continue for second block of layer1
    identity = x
    out = model.layer1[1].conv1(x)
    print(f"After layer1.1.conv1: {out.shape}")
    out = model.layer1[1].bn1(out)
    out = model.layer1[1].relu(out)
    out = model.layer1[1].conv2(out)
    print(f"After layer1.1.conv2: {out.shape}")
    out = model.layer1[1].bn2(out)
    out += identity
    out = model.layer1[1].relu(out)
    x = out
    print(f"After layer1.1 (block): {x.shape}")
    
    # Layer 2 - First block (with downsample)
    identity = x
    out = model.layer2[0].conv1(x)
    print(f"After layer2.0.conv1: {out.shape} (input was {x.shape})")
    out = model.layer2[0].bn1(out)
    out = model.layer2[0].relu(out)
    out = model.layer2[0].conv2(out)
    print(f"After layer2.0.conv2: {out.shape}")
    out = model.layer2[0].bn2(out)
    # Downsample identity
    identity = model.layer2[0].downsample(identity)
    print(f"After layer2.0.downsample: {identity.shape}")
    out += identity
    out = model.layer2[0].relu(out)
    x = out
    print(f"After layer2.0 (block): {x.shape}")
    
    # Second block of layer2
    identity = x
    out = model.layer2[1].conv1(x)
    print(f"After layer2.1.conv1: {out.shape} (input was {x.shape})")

print("\n=== Conv2d Layer Analysis ===")
for i, (name, module) in enumerate(model.named_modules()):
    if isinstance(module, torch.nn.Conv2d):
        print(f"{i}: {name} - in_channels={module.in_channels}, out_channels={module.out_channels}") 