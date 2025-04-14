"""
Benchmark for the Linear layer with different input and output sizes.
"""

import torch
import numpy as np
import logging
import pytest

from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt
from supersayan.nn.layers.linear import Linear

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "batch_size,in_features,out_features",
    [
        (1, 10, 10),
        (1, 10, 100),
        (1, 100, 10),
        (1, 100, 100),
        (8, 10, 10),
        (8, 100, 100),
    ]
)
def test_benchmark_linear(benchmark, batch_size, in_features, out_features):
    """Benchmark Linear layer with different input and output sizes."""
    # Generate secret key
    key = generate_secret_key()

    # Define FHE linear layer
    linear = Linear(in_features=in_features, out_features=out_features, bias=True)

    # Generate random input
    input_data = torch.randn(batch_size, in_features)

    # Encrypt the input data
    encrypted_input = encrypt(input_data, key)

    # Benchmark the forward pass
    result = benchmark(lambda: linear(encrypted_input))

    # Verify shape
    expected_shape = (batch_size, out_features)
    decrypted_output = decrypt(result, key)
    assert decrypted_output.shape == expected_shape