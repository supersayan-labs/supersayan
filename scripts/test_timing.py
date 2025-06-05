#!/usr/bin/env python3
"""
Simple test script to demonstrate detailed timing functionality.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import configure_logging, get_logger
from supersayan.remote.client import SupersayanClient

# Configure logging
configure_logging(
    level="INFO", console_format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = get_logger(__name__)


def test_simple_model_timing(server: str = "127.0.0.1:8000", num_samples: int = 3):
    """Test timing with a simple model."""
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(8, 2)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.linear1(x))
            return self.linear2(x)
    
    # Create model and test data
    model = SimpleModel()
    model.eval()
    
    test_data = SupersayanTensor(
        torch.randn(num_samples, 4),
        device=torch.device("cpu")
    )
    
    print(f"Testing with {num_samples} samples, input shape: {test_data.shape}")
    
    # Create client with Linear layers as FHE modules
    client = SupersayanClient(
        server_url=server,
        torch_model=model,
        fhe_modules=[nn.Linear]  # Both linear layers will be FHE
    )
    
    # Reset timing
    client.reset_timing()
    
    # Run inference
    print("Running inference...")
    output = client(test_data)
    print(f"Output shape: {output.shape}")
    
    # Get timing summary
    timing_summary = client.get_timing_summary()
    
    # Print detailed breakdown
    print("\n" + "="*80)
    print("DETAILED TIMING BREAKDOWN")
    print("="*80)
    
    # FHE Layers
    print(f"\nFHE LAYERS:")
    print(f"{'Layer':<15} {'Encrypt':<10} {'Send':<10} {'Inference':<10} {'Receive':<10} {'Decrypt':<10} {'In Size':<10} {'Out Size':<10}")
    print("-" * 105)
    
    for layer_name, metrics in timing_summary['fhe_layers'].items():
        print(f"{layer_name:<15} "
              f"{metrics['avg_encryption_time']*1000:>8.2f}ms "
              f"{metrics['avg_send_time']*1000:>8.2f}ms "
              f"{metrics['avg_inference_time']*1000:>8.2f}ms "
              f"{metrics['avg_receive_time']*1000:>8.2f}ms "
              f"{metrics['avg_decryption_time']*1000:>8.2f}ms "
              f"{metrics['avg_encrypted_input_size']:>8.0f}B "
              f"{metrics['avg_encrypted_output_size']:>8.0f}B")
    
    # Non-FHE Layers
    if timing_summary['non_fhe_layers']:
        print(f"\nNON-FHE LAYERS:")
        print(f"{'Layer':<15} {'Torch Time':<12}")
        print("-" * 30)
        
        for layer_name, metrics in timing_summary['non_fhe_layers'].items():
            print(f"{layer_name:<15} {metrics['avg_torch_inference_time']*1000:>10.2f}ms")
    
    # Totals
    totals = timing_summary['totals']
    print(f"\nTOTALS (averaged across {totals['total_samples']} samples):")
    print("-" * 50)
    print(f"Total Encryption Time:     {totals['avg_total_encryption_time']*1000:>8.2f}ms")
    print(f"Total Send Time:           {totals['avg_total_send_time']*1000:>8.2f}ms")
    print(f"Total Inference Time:      {totals['avg_total_inference_time']*1000:>8.2f}ms")
    print(f"Total Receive Time:        {totals['avg_total_receive_time']*1000:>8.2f}ms")
    print(f"Total Decryption Time:     {totals['avg_total_decryption_time']*1000:>8.2f}ms")
    print(f"Total Torch Inference:     {totals['avg_total_torch_inference_time']*1000:>8.2f}ms")
    
    total_fhe_time = (totals['avg_total_encryption_time'] + 
                      totals['avg_total_send_time'] + 
                      totals['avg_total_inference_time'] + 
                      totals['avg_total_receive_time'] + 
                      totals['avg_total_decryption_time'])
    total_time = total_fhe_time + totals['avg_total_torch_inference_time']
    
    print(f"Total FHE Pipeline Time:   {total_fhe_time*1000:>8.2f}ms")
    print(f"TOTAL TIME:                {total_time*1000:>8.2f}ms")
    
    print("\n" + "="*80)
    
    return timing_summary


if __name__ == "__main__":
    test_simple_model_timing() 