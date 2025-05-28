import logging
from typing import Union

import numpy as np

from supersayan.logging_config import get_logger

from .bindings import SupersayanTFHE
from .types import KEY, LWE, MU, SIGMA, P, SupersayanTensor

logger = get_logger(__name__)


def encrypt_to_lwes(
    mus: SupersayanTensor, key: KEY, sigma: SIGMA = None
) -> SupersayanTensor:
    """
    Convert a SupersayanTensor to a SupersayanTensor of LWE ciphertexts.
    Can handle arrays of any dimension.

    Args:
        mus: The tensor to encrypt. Can be a SupersayanTensor of any shape.
        key: The secret key
        sigma: The noise parameter (optional)

    Returns:
        SupersayanTensor: A SupersayanTensor of LWE ciphertext with shape:
            - Original shape plus ciphertext dimension at the end
            - e.g., (d1, d2) -> (d1, d2, ciphertext_dim)
    """
    original_shape = mus.shape

    mus_flat = mus.flatten()
    mus_julia = mus_flat.to_julia()

    if sigma is not None:
        encrypted_julia = SupersayanTFHE.Encryption.encrypt_to_lwes(
            mus_julia, key, sigma
        )
    else:
        encrypted_julia = SupersayanTFHE.Encryption.encrypt_to_lwes(mus_julia, key)

    encrypted_tensor = SupersayanTensor._from_julia(encrypted_julia)

    _, ciphertext_dim = encrypted_tensor.shape
    new_shape = (*original_shape, ciphertext_dim)
    encrypted_tensor = encrypted_tensor.reshape(new_shape)

    return encrypted_tensor


def decrypt_from_lwes(
    ciphertexts: SupersayanTensor, key: KEY, p: P = None
) -> SupersayanTensor:
    """
    Convert a SupersayanTensor of LWE ciphertexts to a SupersayanTensor of float32 values.
    Can handle arrays of any dimension where the last dimension is the ciphertext dimension.

    Args:
        ciphertexts: The tensor of LWE ciphertexts to decrypt. Can be a SupersayanTensor of any shape.
        key: The secret key
        p: The precision parameter (optional)

    Returns:
        SupersayanTensor: A SupersayanTensor of float32 values with the original shape
            (without the ciphertext dimension)
    """
    output_shape = ciphertexts.shape[:-1]
    ciphertext_dim = ciphertexts.shape[-1]

    n_elements = np.prod(output_shape)
    ciphertexts_flat = ciphertexts.reshape(n_elements, ciphertext_dim)
    ciphertexts_julia = ciphertexts_flat.to_julia()

    if p is not None:
        decrypted_julia = SupersayanTFHE.Encryption.decrypt_from_lwes(
            ciphertexts_julia, key, p
        )
    else:
        decrypted_julia = SupersayanTFHE.Encryption.decrypt_from_lwes(
            ciphertexts_julia, key
        )

    decrypted_tensor = SupersayanTensor._from_julia(decrypted_julia)

    decrypted_tensor = decrypted_tensor.reshape(output_shape)

    return decrypted_tensor
