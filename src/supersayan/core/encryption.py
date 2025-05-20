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

    Args:
        mus: The numpy array of float32 values to encrypt
        key: The encryption key
        sigma: The noise parameter (optional)

    Returns:
        np.ndarray[LWE]: An numpy array of LWE ciphertext objects
    """
    original_shape = mus.shape

    mus_flattened = mus.flatten()

    if sigma is not None:
        encrypted_flat = SupersayanTFHE.Encryption.encrypt_to_lwes(
            mus_flattened, key, sigma
        )
    else:
        encrypted_flat = SupersayanTFHE.Encryption.encrypt_to_lwes(
            mus_flattened, key
        )
    
    # FIXME: Make sure the reshape is good
    encrypted_np_array = np.array(encrypted_flat)

    encrypted_np_array = [np.array(x).astype(np.float32) for x in encrypted_np_array]
    
    encrypted_np_array = np.array(encrypted_np_array).reshape(original_shape + (-1,))

    return encrypted_np_array

def decrypt_from_lwes(
    ciphertexts: np.ndarray[LWE], key: KEY, p: P = None
) -> np.ndarray[MU]:
    """
    Convert an numpy array of LWE ciphertexts to an numpy array of float32 values.

    Args:
        ciphertexts: The numpy array of LWE ciphertexts to decrypt
        key: The secret key for decryption
        p: Precision parameter for decryption

    Returns:
        np.ndarray[MU]: A numpy array of float32 values with the same shape as the input

    Raises:
        RuntimeError: If Julia decryption fails
    """
    original_shape = ciphertexts.shape

    ciphertexts_flattened = ciphertexts.reshape(-1, ciphertexts.shape[-1])
   
    if p is not None:
        decrypted_values = SupersayanTFHE.Encryption.decrypt_from_lwes(
            ciphertexts_flattened, key, p
        )
    else: 
        decrypted_values = SupersayanTFHE.Encryption.decrypt_from_lwes(
            ciphertexts_flattened, key
        )


    decrypted_np_array = np.array(decrypted_values).reshape(original_shape[:-1])

    return decrypted_np_array

if __name__ == "__main__":
    key = SupersayanTFHE.Encryption.generate_key()
    # mu = np.linspace(0, 1, 11, dtype=np.float32)
    # encrypted = encrypt_to_lwes(mu, key)
    # decrypted = decrypt_from_lwes(encrypted, key)
    # print(mu == decrypted)
    # print(mu)
    # print(decrypted)

    encrypted = encrypt_to_lwes(np.array(0.1), key)
    added = SupersayanTFHE.Operations.add(encrypted, np.float32(0.1))
    decrypted = decrypt_from_lwes(added, key)
    print(decrypted)