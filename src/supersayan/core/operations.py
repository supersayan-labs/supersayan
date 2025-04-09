import logging
import numpy as np
from .bindings import jl
from .types import LWE
from typing import Union

logger = logging.getLogger(__name__)

def add_lwe(lhs: Union[LWE, np.ndarray[LWE]], rhs: Union[float, np.ndarray[LWE]]) -> Union[LWE, np.ndarray[LWE]]:
    """
    Adds LWE ciphertexts.
    
    Args:
        lhs (Union[LWE, np.ndarray[LWE]]): The left-hand side operand, which can be either a single LWE ciphertext or an array of LWE ciphertexts.
        rhs (Union[float, np.ndarray[LWE]]): The right-hand side operand, which can be either a scalar or an array of LWE ciphertexts.
        
    Returns:
        Union[LWE, np.ndarray[LWE]]: The resulting LWE ciphertext or array of LWE ciphertexts after addition.

    Raises:
        RuntimeError: If Julia addition fails
    """
    try:
        return jl.SupersayanTFHE.Operations.add(lhs, rhs)
    except Exception as e:
        logger.error("LWE addition failed: %s", e)
        raise RuntimeError(f"LWE addition failed: {e}") from e

def dot_product_lwe(enc_vec: np.ndarray[LWE], plain_vec: np.ndarray[float]) -> LWE:
    """
    Computes the encrypted dot product between an encrypted vector (list of LWE ciphertexts)
    and a plaintext vector (list of numbers) using the Julia backend.
    
    Args:
        enc_vec (np.ndarray[LWE]): The encrypted vector (list of LWE ciphertexts).
        plain_vec (np.ndarray[float]): The plaintext vector (list of numbers).
    
    Returns:
        LWE: The resulting LWE ciphertext after computing the dot product.
    
    Raises:
        RuntimeError: If Julia dot product fails.
    """
    # Create a zero ciphertext with same mask length as the encrypted vector elements
    mask_length = len(enc_vec[0].mask) if enc_vec else 0
    zero_cipher = LWE(np.zeros(mask_length, dtype=np.float64), 0.0)
    
    try:
        result = jl.SupersayanTFHE.Operations.dot_product(enc_vec, plain_vec, zero_cipher)
        return LWE.from_julia(result)
    except Exception as e:
        logger.error("Encrypted dot product failed: %s", e)
        raise RuntimeError(f"Encrypted dot product failed: {e}") from e
