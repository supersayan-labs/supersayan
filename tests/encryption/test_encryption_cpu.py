import numpy as np
import pytest

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import configure_logging, get_logger

configure_logging(level="INFO")
logger = get_logger(__name__)

np.random.seed(42)

EPSILON = 0.1


@pytest.fixture(scope="module")
def fhe_secret_key():
    """Generate secret key fixture for FHE operations."""
    logger.info("Generating secret key (fixture)...")
    return generate_secret_key()


def test_encryption_decryption_round_trip_cpu(fhe_secret_key):
    """Test that encryption followed by decryption returns the original values within epsilon."""

    test_cases = [
        SupersayanTensor(np.linspace(0, 0.99, 11, dtype=np.float32)),
        SupersayanTensor(np.random.random((3, 4)).astype(np.float32)),
        SupersayanTensor(np.random.random((2, 3, 4)).astype(np.float32)),
    ]

    for original in test_cases:
        encrypted = encrypt_to_lwes(original, fhe_secret_key)

        decrypted = decrypt_from_lwes(encrypted, fhe_secret_key)

        assert (
            original.shape == decrypted.shape
        ), f"Shape mismatch: {original.shape} vs {decrypted.shape}"

        max_delta = np.max(np.abs(original.to_numpy() - decrypted.to_numpy()))
        print(f"Max delta for shape {original.shape}: {max_delta}")

        assert max_delta <= EPSILON, f"Max delta {max_delta} exceeds epsilon {EPSILON}"


def test_large_array_cpu(fhe_secret_key):
    """Test with a larger array to ensure performance and memory handling."""
    original = SupersayanTensor(np.random.random((10, 10)).astype(np.float32))

    encrypted = encrypt_to_lwes(original, fhe_secret_key)
    decrypted = decrypt_from_lwes(encrypted, fhe_secret_key)

    max_delta = np.max(np.abs(original.to_numpy() - decrypted.to_numpy()))
    print(f"Max delta for large array: {max_delta}")

    assert max_delta <= EPSILON, f"Max delta {max_delta} exceeds epsilon {EPSILON}"
