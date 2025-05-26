import torch.nn as nn

from .linear import Linear
from .conv2d import Conv2d

LAYER_MAPPING = {nn.Linear: Linear, nn.Conv2d: Conv2d}
