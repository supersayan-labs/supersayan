import pytest
import numpy as np
from supersayan.core.encryption import encrypt_to_lwes, decrypt_from_lwes
from supersayan.core.keygen import generate_secret_key

# Set a fixed seed for reproducible results
np.random.seed(42)

EPSILON = 0.1


def test_encryption_decryption_round_trip():
    """Test that encryption followed by decryption returns the original values within epsilon."""
    # Generate encryption key
    key = generate_secret_key()

    # Test with various shapes
    test_cases = [
        np.linspace(0, 0.99, 11, dtype=np.float32),  # 1D array
        np.random.random((3, 4)).astype(np.float32),  # 2D array
        np.random.random((2, 3, 4)).astype(np.float32),  # 3D array
    ]

    for original in test_cases:
        # Encrypt the original values
        encrypted = encrypt_to_lwes(original, key)

        # Decrypt the encrypted values
        decrypted = decrypt_from_lwes(encrypted, key)

        # Check shapes
        assert (
            original.shape == decrypted.shape
        ), f"Shape mismatch: {original.shape} vs {decrypted.shape}"

        # Calculate max delta
        max_delta = np.max(np.abs(original - decrypted))
        print(f"Max delta for shape {original.shape}: {max_delta}")

        # Check values are close within epsilon
        assert max_delta <= EPSILON, f"Max delta {max_delta} exceeds epsilon {EPSILON}"


def test_large_array():
    """Test with a larger array to ensure performance and memory handling."""
    key = generate_secret_key()
    original = np.random.random((10, 10)).astype(np.float32)

    encrypted = encrypt_to_lwes(original, key)
    decrypted = decrypt_from_lwes(encrypted, key)

    # Calculate max delta
    max_delta = np.max(np.abs(original - decrypted))
    print(f"Max delta for large array: {max_delta}")

    assert max_delta <= EPSILON, f"Max delta {max_delta} exceeds epsilon {EPSILON}"


if __name__ == "__main__":
    test_encryption_decryption_round_trip()
    test_large_array()
