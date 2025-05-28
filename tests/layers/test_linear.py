import numpy as np
import pytest
import torch

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import configure_logging, get_logger
from supersayan.nn.layers.linear import Linear

configure_logging(level="INFO")
logger = get_logger(__name__)


@pytest.fixture(scope="module")
def fhe_secret_key():
    """Generate secret key fixture for FHE operations."""
    logger.info("Generating secret key (fixture)...")
    return generate_secret_key()


def test_linear_layer(fhe_secret_key):
    """Test FHE Linear layer forward pass."""
    logger.info("Creating Linear layer...")
    linear = Linear(in_features=8, out_features=4, bias=True)

    input_data = SupersayanTensor(torch.rand(1, 8), device=torch.device("cuda"))
    print("Input shape:", input_data.shape)

    try:
        logger.info("Encrypting input data...")
        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)

        logger.info("Processing linear layer...")
        output_encrypted = linear(encrypted_input)

        logger.info("Decrypting result...")
        decrypted_output = decrypt_from_lwes(output_encrypted, fhe_secret_key)

        logger.info(f"Output shape: {decrypted_output.shape}")

    except Exception as e:
        logger.error(f"FHE linear failed: {e}")
        raise

    expected_shape = (1, 4)
    assert (
        decrypted_output.shape == expected_shape
    ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

    logger.info("Linear test completed successfully")
