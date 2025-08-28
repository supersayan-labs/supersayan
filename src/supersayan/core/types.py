from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack

from supersayan.core.bindings import HAS_CUPY, cp, jl
from supersayan.logging_config import get_logger

logger = get_logger(__name__)

A = np.ndarray[np.float32]
B = np.float32
LWE = np.ndarray[np.float32]
MU = np.float32
SIGMA = np.float32
KEY = np.ndarray[np.float32]
P = np.int32


def extract_lwe(x: LWE) -> Tuple[A, B]:
    return x[1:], x[0]


def pack_lwe(x: A, y: B) -> LWE:
    return np.concatenate(([y], x))


class SupersayanTensor(torch.Tensor):
    """
    A wrapper around torch.Tensor that abstracts CPU/GPU & numpy/cupy/torch conversions with the Julia backend.

    This class provides zero-copy array sharing between PyTorch, NumPy, CuPy, and Julia.
    """

    @staticmethod
    def __new__(
        cls,
        data: Union[torch.Tensor, np.ndarray, "cp.ndarray", Any],
        device: torch.device = torch.device("cpu"),
    ) -> "SupersayanTensor":
        """
        Create a new SupersayanTensor from various array types.

        Args:
            data: The input data to convert to a SupersayanTensor
            device: The device to place the tensor on

        Returns:
            SupersayanTensor: A new SupersayanTensor instance
        """
        if data.dtype == torch.Tensor and data.device != device:
            raise ValueError(f"Device mismatch: {data.device} != {device}")

        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)

            if device is not None:
                tensor = tensor.to(device)
        elif HAS_CUPY and isinstance(data, cp.ndarray):
            tensor = torch_dlpack.from_dlpack(data.toDlpack())
        else:
            raise ValueError(f"Unsupported tensor type: {type(data)}")

        if tensor.dtype != torch.float32:
            logger.warning(
                f"Tensor type unsupported, converting to float32: {tensor.dtype} -> float32"
            )
            tensor = tensor.to(dtype=torch.float32)

        instance = tensor.as_subclass(cls)

        return instance

    @classmethod
    def _from_julia(cls, julia_array: Any) -> "SupersayanTensor":
        """
        Convert a Julia array to SupersayanTensor.

        Args:
            julia_array: The Julia array to convert to a SupersayanTensor

        Returns:
            SupersayanTensor: A new SupersayanTensor instance
        """
        jl.temp_array = julia_array

        is_cuda = bool(jl.seval("isa(temp_array, CUDA.CuArray)"))

        if is_cuda:
            if not HAS_CUPY:
                raise RuntimeError("CuPy required for CUDA arrays but not installed")

            jl.seval("temp_cupy = DLPack.share(temp_array, cupy.from_dlpack)")
            cupy_array = jl.temp_cupy
            tensor = torch_dlpack.from_dlpack(cupy_array.toDlpack())

            if tensor.ndim > 1:
                axes = list(reversed(range(tensor.ndim)))
                tensor = tensor.permute(axes).contiguous()
        else:
            tensor = np.asarray(julia_array)
            tensor = torch.from_numpy(tensor)

        return tensor.as_subclass(cls)

    def to_julia(self) -> Any:
        """
        Convert to Julia array with zero-copy.

        Returns:
            Any: The Julia array
        """
        axes = list(reversed(range(self.ndim)))
        jl.temp_tensor = self.permute(axes).contiguous()

        jl.seval("temp_julia = from_dlpack(temp_tensor)")

        return jl.temp_julia

    def to_numpy(self) -> np.ndarray:
        """
        Convert to NumPy array with zero-copy.

        Returns:
            np.ndarray: The NumPy array
        """
        if self.is_cuda:
            return self.detach().cpu().numpy()

        return self.detach().numpy()

    @property
    def is_cuda(self) -> bool:
        """
        Check if tensor is on CUDA device.

        Returns:
            bool: True if tensor is on CUDA device, False otherwise
        """
        return self.device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """
        Check if tensor is on CPU.

        Returns:
            bool: True if tensor is on CPU, False otherwise
        """
        return self.device.type == "cpu"

    def to_dlpack(self) -> Any:
        """
        Export tensor via DLPack interface.

        Returns:
            Any: The DLPack object
        """
        return torch_dlpack.to_dlpack(self)

    @classmethod
    def from_dlpack(cls, dlpack_obj: Any) -> "SupersayanTensor":
        """
        Create SupersayanTensor from DLPack object.

        Args:
            dlpack_obj: The DLPack object to create a SupersayanTensor from

        Returns:
            SupersayanTensor: A new SupersayanTensor instance
        """
        tensor = torch_dlpack.from_dlpack(dlpack_obj)

        return cls(tensor)

    @classmethod
    def zeros(
        cls, *size, device: Optional[Union[str, torch.device]] = None, **kwargs
    ) -> "SupersayanTensor":
        """
        Create a zero-filled SupersayanTensor.

        Args:
            size: The size of the tensor
            device: The device to place the tensor on

        Returns:
            SupersayanTensor: A new SupersayanTensor instance
        """
        tensor = torch.zeros(*size, dtype=torch.float32, device=device, **kwargs)
        return cls(tensor)

    @classmethod
    def ones(
        cls, *size, device: Optional[Union[str, torch.device]] = None, **kwargs
    ) -> "SupersayanTensor":
        """
        Create a one-filled SupersayanTensor.

        Args:
            size: The size of the tensor
            device: The device to place the tensor on

        Returns:
            SupersayanTensor: A new SupersayanTensor instance
        """
        tensor = torch.ones(*size, dtype=torch.float32, device=device, **kwargs)
        return cls(tensor)

    @classmethod
    def randn(
        cls, *size, device: Optional[Union[str, torch.device]] = None, **kwargs
    ) -> "SupersayanTensor":
        """
        Create a SupersayanTensor with random normal values.

        Args:
            size: The size of the tensor
            device: The device to place the tensor on

        Returns:
            SupersayanTensor: A new SupersayanTensor instance
        """
        tensor = torch.randn(*size, dtype=torch.float32, device=device, **kwargs)
        return cls(tensor)

    def __repr__(self) -> str:
        """
        String representation of SupersayanTensor.

        Returns:
            str: The string representation
        """
        return super().__repr__().replace("tensor", "SupersayanTensor", 1)
