import torch.nn as nn

from .conv2d import Conv2d
from .gelu import GELU
from .linear import Linear

LAYER_MAPPING = {nn.Linear: Linear, nn.Conv2d: Conv2d, nn.ReLU: GELU}
