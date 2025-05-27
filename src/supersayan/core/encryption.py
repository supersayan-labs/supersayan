import logging
import numpy as np
from .bindings import SupersayanTFHE
from .types import LWE, SIGMA, MU, KEY, P

logger = logging.getLogger(__name__)


def encrypt_to_lwes(
    mus: np.ndarray[MU], key: KEY, sigma: SIGMA = None
) -> np.ndarray[LWE]:
    """
    Convert an numpy array of float32 values to an numpy array of LWE ciphertexts.
    Can handle both single arrays and batch inputs.

    Args:
        mus: The numpy array of float32 values to encrypt. Can be:
            - Single array: shape (d1, d2, ..., dn)
            - Batch of arrays: shape (batch_size, d1, d2, ..., dn)
        key: The secret key
        sigma: The noise parameter (optional)

    Returns:
        np.ndarray[LWE]: An numpy array of LWE ciphertext with shape:
            - Single array: (..., ciphertext_dim)
            - Batch: (batch_size, ..., ciphertext_dim)
    """
    if mus.ndim > 0:
        batch_size = mus.shape[0]
        encrypted_batch = []
        
        for i in range(batch_size):
            encrypted_single = encrypt_to_lwes(mus[i], key, sigma)
            encrypted_batch.append(encrypted_single)
        
        return np.asarray(encrypted_batch)
    
    original_shape = mus.shape

    mus_flattened = mus.flatten()

    if sigma is not None:
        encrypted_flat = SupersayanTFHE.Encryption.encrypt_to_lwes(
            mus_flattened, key, sigma
        )
    else:
        encrypted_flat = SupersayanTFHE.Encryption.encrypt_to_lwes(mus_flattened, key)

    # FIXME: Make sure the reshape is good
    encrypted_np_array = np.asarray(encrypted_flat)

    encrypted_np_array = np.asarray([np.asarray(x).astype(np.float32) for x in encrypted_np_array])
    
    encrypted_np_array = encrypted_np_array.reshape((*original_shape, -1))

    return encrypted_np_array


def decrypt_from_lwes(
    ciphertexts: np.ndarray[LWE], key: KEY, p: P = None
) -> np.ndarray[MU]:
    """
    Convert an numpy array of LWE ciphertexts to an numpy array of float32 values.
    Can handle both single arrays and batch inputs.

    Args:
        ciphertexts: The numpy array of LWE ciphertexts to decrypt. Can be:
            - Single array: shape (..., ciphertext_dim)
            - Batch of arrays: shape (batch_size, ..., ciphertext_dim)
        key: The secret key
        p: The precision parameter (optional)

    Returns:
        np.ndarray[MU]: A numpy array of float32 values with shape:
            - Single array: original shape without ciphertext_dim
            - Batch: (batch_size, ...) without ciphertext_dim
    """
    if ciphertexts.ndim > 1:
        batch_size = ciphertexts.shape[0]
        decrypted_batch = []
        
        for i in range(batch_size):
            decrypted_single = decrypt_from_lwes(ciphertexts[i], key, p)
            decrypted_batch.append(decrypted_single)
        
        return np.asarray(decrypted_batch)
    
    ciphertexts_np = np.asarray(ciphertexts)

    original_shape = ciphertexts_np.shape

    ciphertexts_flattened = np.reshape(ciphertexts_np, (-1, ciphertexts_np.shape[-1]))

    if p is not None:
        decrypted_values = SupersayanTFHE.Encryption.decrypt_from_lwes(
            ciphertexts_flattened, key, p
        )
    else:
        decrypted_values = SupersayanTFHE.Encryption.decrypt_from_lwes(
            ciphertexts_flattened, key
        )

    decrypted_np_array = np.asarray(decrypted_values).reshape(original_shape[:-1])

    return decrypted_np_array
