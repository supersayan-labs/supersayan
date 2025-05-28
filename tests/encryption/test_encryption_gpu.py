import numpy as np
import pytest

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.core.types import SupersayanTensor
from supersayan.core.bindings import cp
from supersayan.logging_config import configure_logging, get_logger

import torch

configure_logging(level="INFO")
logger = get_logger(__name__)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
skip_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

np.random.seed(42)

EPSILON = 0.1

@pytest.fixture(scope="module")
def fhe_secret_key():
    """Generate secret key fixture for FHE operations."""
    logger.info("Generating secret key (fixture)...")
    return generate_secret_key()


@skip_cuda
def test_encryption_decryption_round_trip_gpu(fhe_secret_key):
    """Test that encryption followed by decryption returns the original values within epsilon."""
    test_cases = [
        SupersayanTensor(cp.linspace(0, 0.99, 11, dtype=np.float32), device=torch.device("cuda")),
        SupersayanTensor(cp.random.random((3, 4)).astype(np.float32), device=torch.device("cuda")),
        SupersayanTensor(cp.random.random((2, 3, 4)).astype(np.float32), device=torch.device("cuda")),
    ]

    for i, original in enumerate(test_cases):
        print(f"\nTest case {i+1}, shape: {original.shape}")
        print(f"Original values sample: {original.flatten()[:5].cpu().numpy()}")
        
        encrypted = encrypt_to_lwes(original, fhe_secret_key)
        print(f"Encrypted shape: {encrypted.shape}")

        decrypted = decrypt_from_lwes(encrypted, fhe_secret_key)
        print(f"Decrypted values sample: {decrypted.flatten()[:5].cpu().numpy()}")

        assert (
            original.shape == decrypted.shape
        ), f"Shape mismatch: {original.shape} vs {decrypted.shape}"

        max_delta = np.max(np.abs(original.to_numpy() - decrypted.to_numpy()))
        print(f"Max delta for shape {original.shape}: {max_delta}")

        assert max_delta <= EPSILON, f"Max delta {max_delta} exceeds epsilon {EPSILON}"


@skip_cuda
def test_large_array_gpu(fhe_secret_key):
    """Test with a larger array to ensure performance and memory handling."""
    original = SupersayanTensor(cp.random.random((10, 10)).astype(np.float32), device=torch.device("cuda"))
    
    print(f"\nLarge array test:")
    print(f"Original shape: {original.shape}")
    print(f"Original values sample: {original.flatten()[:5].cpu().numpy()}")
    print(f"Original min/max: {original.min().item():.6f} / {original.max().item():.6f}")

    encrypted = encrypt_to_lwes(original, fhe_secret_key)
    print(f"Encrypted shape: {encrypted.shape}")
    
    decrypted = decrypt_from_lwes(encrypted, fhe_secret_key)
    print(f"Decrypted values sample: {decrypted.flatten()[:5].cpu().numpy()}")
    print(f"Decrypted min/max: {decrypted.min().item():.6f} / {decrypted.max().item():.6f}")

    max_delta = np.max(np.abs(original.to_numpy() - decrypted.to_numpy()))
    print(f"Max delta for large array: {max_delta}")
    
    # Find where the max delta occurs
    diff = np.abs(original.to_numpy() - decrypted.to_numpy())
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Max delta at index {max_idx}: original={original[max_idx].item():.6f}, decrypted={decrypted[max_idx].item():.6f}")

    assert max_delta <= EPSILON, f"Max delta {max_delta} exceeds epsilon {EPSILON}"