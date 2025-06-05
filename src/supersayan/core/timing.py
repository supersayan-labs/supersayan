from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn

from supersayan.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FHELayerTiming:
    """Timing metrics for FHE layers."""
    layer_name: str
    encryption_time: float = 0.0
    encrypted_input_size_bytes: int = 0
    send_time: float = 0.0
    inference_time: float = 0.0  # From server response
    receive_time: float = 0.0
    encrypted_output_size_bytes: int = 0
    decryption_time: float = 0.0


@dataclass
class NonFHELayerTiming:
    """Timing metrics for non-FHE layers."""
    layer_name: str
    torch_inference_time: float = 0.0


@dataclass
class SampleTiming:
    """Timing metrics for a single sample."""
    sample_id: int
    fhe_layers: List[FHELayerTiming] = field(default_factory=list)
    non_fhe_layers: List[NonFHELayerTiming] = field(default_factory=list)
    total_time: float = 0.0


class TimingCollector:
    """Collects timing metrics for benchmarking."""
    
    def __init__(self):
        self.samples: List[SampleTiming] = []
        self.current_sample: Optional[SampleTiming] = None
        self.sample_counter = 0
        
    def start_sample(self) -> int:
        """Start timing a new sample."""
        self.sample_counter += 1
        self.current_sample = SampleTiming(sample_id=self.sample_counter)
        logger.debug(f"Started timing sample {self.sample_counter}")
        return self.sample_counter
        
    def end_sample(self) -> None:
        """End timing the current sample."""
        if self.current_sample is not None:
            self.samples.append(self.current_sample)
            logger.debug(f"Ended timing sample {self.current_sample.sample_id}")
            self.current_sample = None
    
    def add_fhe_layer_timing(self, timing: FHELayerTiming) -> None:
        """Add FHE layer timing to current sample."""
        if self.current_sample is not None:
            self.current_sample.fhe_layers.append(timing)
            logger.debug(f"Added FHE timing for layer {timing.layer_name}")
    
    def add_non_fhe_layer_timing(self, timing: NonFHELayerTiming) -> None:
        """Add non-FHE layer timing to current sample."""
        if self.current_sample is not None:
            self.current_sample.non_fhe_layers.append(timing)
            logger.debug(f"Added non-FHE timing for layer {timing.layer_name}")
    
    def clear(self) -> None:
        """Clear all collected timings."""
        self.samples.clear()
        self.current_sample = None
        self.sample_counter = 0
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all samples."""
        if not self.samples:
            return {}
        
        stats = {
            "total_samples": len(self.samples),
            "fhe_layers": {},
            "non_fhe_layers": {},
        }
        
        # Collect FHE layer stats
        fhe_layer_names = set()
        for sample in self.samples:
            fhe_layer_names.update(timing.layer_name for timing in sample.fhe_layers)
        
        for layer_name in fhe_layer_names:
            layer_timings = []
            for sample in self.samples:
                for timing in sample.fhe_layers:
                    if timing.layer_name == layer_name:
                        layer_timings.append(timing)
            
            if layer_timings:
                encryption_times = [t.encryption_time for t in layer_timings]
                send_times = [t.send_time for t in layer_timings]
                inference_times = [t.inference_time for t in layer_timings]
                receive_times = [t.receive_time for t in layer_timings]
                decryption_times = [t.decryption_time for t in layer_timings]
                input_sizes = [t.encrypted_input_size_bytes for t in layer_timings]
                output_sizes = [t.encrypted_output_size_bytes for t in layer_timings]
                
                stats["fhe_layers"][layer_name] = {
                    "count": len(layer_timings),
                    "encryption_time": {
                        "mean": sum(encryption_times) / len(encryption_times),
                        "min": min(encryption_times),
                        "max": max(encryption_times),
                    },
                    "send_time": {
                        "mean": sum(send_times) / len(send_times),
                        "min": min(send_times),
                        "max": max(send_times),
                    },
                    "inference_time": {
                        "mean": sum(inference_times) / len(inference_times),
                        "min": min(inference_times),
                        "max": max(inference_times),
                    },
                    "receive_time": {
                        "mean": sum(receive_times) / len(receive_times),
                        "min": min(receive_times),
                        "max": max(receive_times),
                    },
                    "decryption_time": {
                        "mean": sum(decryption_times) / len(decryption_times),
                        "min": min(decryption_times),
                        "max": max(decryption_times),
                    },
                    "encrypted_input_size_bytes": {
                        "mean": sum(input_sizes) / len(input_sizes),
                        "min": min(input_sizes),
                        "max": max(input_sizes),
                    },
                    "encrypted_output_size_bytes": {
                        "mean": sum(output_sizes) / len(output_sizes),
                        "min": min(output_sizes),
                        "max": max(output_sizes),
                    },
                }
        
        # Collect non-FHE layer stats
        non_fhe_layer_names = set()
        for sample in self.samples:
            non_fhe_layer_names.update(timing.layer_name for timing in sample.non_fhe_layers)
        
        for layer_name in non_fhe_layer_names:
            layer_timings = []
            for sample in self.samples:
                for timing in sample.non_fhe_layers:
                    if timing.layer_name == layer_name:
                        layer_timings.append(timing)
            
            if layer_timings:
                torch_times = [t.torch_inference_time for t in layer_timings]
                
                stats["non_fhe_layers"][layer_name] = {
                    "count": len(layer_timings),
                    "torch_inference_time": {
                        "mean": sum(torch_times) / len(torch_times),
                        "min": min(torch_times),
                        "max": max(torch_times),
                    },
                }
        
        return stats


def is_leaf_module(module: nn.Module) -> bool:
    """Check if a module is a leaf module (has no child modules)."""
    return len(list(module.children())) == 0


def get_object_size_bytes(obj: Any) -> int:
    """Get the size of an object in bytes."""
    try:
        import pickle
        return len(pickle.dumps(obj))
    except Exception:
        # Fallback to sys.getsizeof if pickle fails
        return sys.getsizeof(obj)


# Global timing collector instance
_timing_collector = TimingCollector()


def get_timing_collector() -> TimingCollector:
    """Get the global timing collector instance."""
    return _timing_collector


def reset_timing_collector() -> None:
    """Reset the global timing collector."""
    global _timing_collector
    _timing_collector = TimingCollector() 