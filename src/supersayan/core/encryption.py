import logging
import numpy as np
import torch
from typing import List, Union, Optional, overload
from .bindings import SupersayanTFHE
from .types import LWE

logger = logging.getLogger(__name__)

@overload
def encrypt(mu: float, key: List[float], sigma: Optional[float] = None) -> LWE: ...

@overload
def encrypt(mu: torch.Tensor, key: List[float], sigma: Optional[float] = None) -> np.ndarray: ...

def encrypt(mu: Union[float, torch.Tensor], key: List[float], sigma: Optional[float] = None) -> Union[LWE, np.ndarray]:
    """
    Encrypt a single torus value or every element in a PyTorch tensor.
    
    Args:
        mu: The torus value or PyTorch tensor to encrypt
        key: The encryption key
        sigma: Optional noise parameter
        
    Returns:
        Union[LWE, np.ndarray]: An LWE ciphertext object or a numpy array of LWE ciphertext objects
        
    Raises:
        RuntimeError: If Julia encryption fails
    """
    # Handle single float value
    if isinstance(mu, float) or isinstance(mu, int):
        try:
            if sigma is not None:
                result = SupersayanTFHE.Encryption.encrypt_torus_to_lwe(float(mu), key, sigma)
            else:
                result = SupersayanTFHE.Encryption.encrypt_torus_to_lwe(float(mu), key)
            return LWE.from_julia(result)
        except Exception as e:
            logger.error(f"Julia encryption failed for single value: {e}")
            raise RuntimeError(f"Encryption failed: {e}") from e
    
    # Handle tensor
    elif isinstance(mu, torch.Tensor):
        original_shape = mu.shape
        
        # Convert to numpy array, flatten to 1D, and ensure it's float64
        mu_np_flat = mu.detach().cpu().numpy().flatten().astype(np.float64)
        
        # Encrypt using Julia backend (works with 1D array only)
        try:
            if sigma is not None:
                encrypted_flat = SupersayanTFHE.Encryption.encrypt_torus_to_lwe_vec(mu_np_flat, key, sigma)
            else:
                encrypted_flat = SupersayanTFHE.Encryption.encrypt_torus_to_lwe_vec(mu_np_flat, key)
        except Exception as e:
            logger.error(f"Julia encryption failed for tensor: {e}")
            raise RuntimeError(f"Encryption failed: {e}") from e

        # Convert Julia LWE objects to Python LWE objects using the batch conversion method
        python_lwe_objects = LWE.from_julia_batch(encrypted_flat)
        
        # Reshape to original dimensions
        encrypted_np_array = python_lwe_objects.reshape(original_shape)
        
        return encrypted_np_array
    
    # Handle unsupported input type
    else:
        raise TypeError(f"Cannot encrypt object of type {type(mu)}. Expected float or torch.Tensor.")

@overload
def decrypt(ciphertext: LWE, key: List[float], p: int = 5) -> float: ...

@overload
def decrypt(ciphertext: np.ndarray, key: List[float], p: int = 5) -> np.ndarray: ...

def decrypt(ciphertext: Union[LWE, np.ndarray], key: List[float], p: int = 5) -> Union[float, np.ndarray]:
    """
    Decrypts LWE ciphertext(s) using the provided secret key.
    
    Args:
        ciphertext: Either a single LWE object or a numpy array of LWE objects
        key: The secret key for decryption
        p: Precision parameter for decryption
        
    Returns:
        Union[float, np.ndarray]: A single float (if input was a single LWE object) or 
                                a numpy array of floats with the same shape as the input
                                
    Raises:
        RuntimeError: If Julia decryption fails
    """
    # Handle single LWE object
    if isinstance(ciphertext, LWE):
        try:
            return SupersayanTFHE.Encryption.decrypt_lwe_to_torus(ciphertext, key, p)
        except Exception as e:
            logger.error(f"Julia decryption failed for single LWE: {e}")
            raise RuntimeError(f"Decryption failed: {e}") from e
        
    # Handle numpy array (any dimension)
    elif isinstance(ciphertext, np.ndarray):
        original_shape = ciphertext.shape
        # Flatten the array for Julia
        flat_ciphertexts = ciphertext.flatten()
            
        try:
            # Decrypt the flattened array
            decrypted_values = SupersayanTFHE.Encryption.decrypt_lwe_to_torus_vec(flat_ciphertexts, key, p)
        except Exception as e:
            logger.error(f"Julia decryption failed for array: {e}")
            raise RuntimeError(f"Decryption failed: {e}") from e
        
        # Reshape back to original dimensions
        return np.array(decrypted_values).reshape(original_shape)
    
    # Handle unsupported input type
    else:
        raise TypeError(f"Cannot decrypt object of type {type(ciphertext)}. Expected LWE or numpy.ndarray.")