import numpy as np
from typing import List, Union

class LWE:
    """
    Python representation of the Julia LWE struct.
    
    An LWE ciphertext consists of a mask (a vector of Float64) and a masked value (Float64).
    """
    def __init__(self, mask: Union[List[float], np.ndarray], masked: float):
        """
        Initialize an LWE ciphertext.
        
        Args:
            mask: The vector of floats representing the mask
            masked: The masked value (float)
        """
        self.mask = mask if isinstance(mask, np.ndarray) else np.array(mask, dtype=np.float64)
        self.masked = float(masked)
    
    @classmethod
    def from_julia(cls, julia_lwe):
        """
        Create an LWE instance from a Julia LWE object.
        
        Args:
            julia_lwe: The Julia LWE object
            
        Returns:
            LWE: A Python LWE instance
        """
        return cls(
            mask=np.array(julia_lwe.mask, dtype=np.float64),
            masked=float(julia_lwe.masked)
        )
        
    @classmethod
    def from_julia_batch(cls, julia_lwe_list):
        """
        Efficiently convert a batch of Julia LWE objects to Python LWE instances.
        
        Args:
            julia_lwe_list: List of Julia LWE objects
            
        Returns:
            np.ndarray: A numpy array of Python LWE instances
        """
        # Vectorized version for batch conversion
        return np.array([cls.from_julia(lwe) for lwe in julia_lwe_list], dtype=object)
    
    def to_dict(self):
        """
        Convert the LWE object to a dictionary for serialization.
        
        Returns:
            dict: A dictionary representation of the LWE object
        """
        return {
            "mask": self.mask.tolist(),
            "masked": self.masked,
            "type": "LWE"
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create an LWE instance from a dictionary.
        
        Args:
            data: Dictionary with mask and masked values
            
        Returns:
            LWE: A Python LWE instance
        """
        if data.get("type") != "LWE":
            raise ValueError(f"Expected LWE type, got {data.get('type')}")
        
        return cls(
            mask=data["mask"],
            masked=data["masked"]
        )
    
    def __repr__(self):
        return f"LWE(mask_size={len(self.mask)}, masked={self.masked})"