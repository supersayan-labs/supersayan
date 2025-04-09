import torch
import torch.nn as nn
import numpy as np
from typing import List
import logging

from supersayan.core.operations import dot_product_lwe, add_lwe
from supersayan.core.types import LWE

logger = logging.getLogger(__name__)

class Linear(nn.Module):
    """
    A PyTorch-style Linear layer redefined for encrypted data.
    
    Instead of performing the normal dot product between input and weights,
    it uses homomorphic operations provided by the Julia backend.
    
    The forward pass expects a NumPy array of LWE ciphertexts of shape (batch, in_features).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias as torch Tensors.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input: np.ndarray[LWE]) -> np.ndarray[LWE]:
        """
        Performs the forward pass using homomorphic operations.
        
        Args:
            input (np.ndarray[LWE]): A NumPy array of LWE ciphertexts with shape (batch, in_features).
            
        Returns:
            np.ndarray[LWE]: A NumPy array of LWE ciphertexts with shape (batch, out_features).
        """
        batch_size = input.shape[0]
        output = []
        
        # Detach the weight and bias parameters as plain numbers.
        weight_np = self.weight.detach().cpu().numpy()  # shape: (out_features, in_features)
        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None
        
        # For each sample in the batch.
        for i in range(batch_size):
            sample_out = []
            # Each sample is assumed to be an array (or list) of LWE objects of length in_features.
            sample_input = input[i]
            # Ensure sample_input is a list.
            sample_input_list = sample_input.tolist() if isinstance(sample_input, np.ndarray) else list(sample_input)
            
            # For each output neuron.
            for j in range(self.out_features):
                # Get the j-th row of weights as a list of floats.
                plain_weight = weight_np[j].tolist()
                # Compute the homomorphic dot product.
                dp = dot_product_lwe(sample_input_list, plain_weight)
                # Add the bias if available.
                if bias_np is not None:
                    # add_lwe expects an array (or list) of LWE objects; here we wrap dp in a list
                    dp = add_lwe(np.array([dp], dtype=object), bias_np[j])[0]
                sample_out.append(dp)
            output.append(sample_out)
        
        # Convert the output list into a NumPy array of objects.
        return np.array(output, dtype=object)
    
    def __repr__(self):
        return f"EncryptedLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
