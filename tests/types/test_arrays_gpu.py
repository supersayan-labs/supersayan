import numpy as np
import pytest
import torch
import torch.utils.dlpack as torch_dlpack

from supersayan.core.bindings import HAS_CUPY, cp, jl
from supersayan.core.types import SupersayanTensor

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
skip_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

skip_cupy = pytest.mark.skipif(
    not HAS_CUPY or not cuda_available,
    reason="CuPy not available or CUDA not available",
)


@skip_cuda
def test_pytorch_interop_gpu():
    """Test PyTorch tensor interoperability on GPU."""
    # Create from PyTorch CUDA tensor
    pt_tensor = torch.linspace(0, 1, 11, dtype=torch.float32, device="cuda")
    st = SupersayanTensor(pt_tensor, device=torch.device("cuda"))

    # Test device property
    assert st.is_cuda
    assert not st.is_cpu
    assert st.device.type == "cuda"

    # Test that operations preserve device
    st2 = st + 2.0
    assert isinstance(st2, SupersayanTensor)
    assert st2.is_cuda
    assert torch.allclose(st2, pt_tensor + 2.0)

    # Test conversion to CPU for numpy
    np_back = st.to_numpy()
    assert np.allclose(np_back, torch.linspace(0, 1, 11).numpy())


@skip_cupy
def test_cupy_interop():
    """Test CuPy array interoperability."""
    # Create from CuPy array
    cp_arr = cp.linspace(0, 1, 11, dtype=cp.float32)
    st = SupersayanTensor(cp_arr)

    # Test device property
    assert st.is_cuda
    assert not st.is_cpu

    # Test DLPack conversion
    dlpack = st.to_dlpack()
    cp_back = cp.from_dlpack(dlpack)
    assert cp.allclose(cp_arr, cp_back)

    # Test operations
    st2 = st + 2.0
    assert isinstance(st2, SupersayanTensor)
    assert st2.is_cuda


@skip_cuda
def test_cpu_gpu_transfer():
    """Test transfers between CPU and GPU."""
    # Create CPU tensor
    cpu_data = torch.randn(5, 5, dtype=torch.float32)
    st_cpu = SupersayanTensor(cpu_data)
    assert st_cpu.is_cpu

    # Transfer to GPU
    st_gpu = SupersayanTensor(st_cpu.cuda())
    assert st_gpu.is_cuda
    assert torch.allclose(st_cpu, st_gpu.cpu())

    # Create directly on GPU
    gpu_data = torch.randn(5, 5, dtype=torch.float32, device="cuda")
    st_gpu2 = SupersayanTensor(gpu_data)
    assert st_gpu2.is_cuda

    # Transfer to CPU
    st_cpu2 = SupersayanTensor(st_gpu2.cpu())
    assert st_cpu2.is_cpu
    assert torch.allclose(st_gpu2.cpu(), st_cpu2)


@skip_cuda
def test_mixed_device_operations():
    """Test operations between tensors on different devices."""
    # Create tensors on different devices
    st_cpu = SupersayanTensor.ones(5, 5)
    st_gpu = SupersayanTensor.ones(5, 5, device="cuda")

    assert st_cpu.is_cpu
    assert st_gpu.is_cuda

    # Operations should handle device transfer (PyTorch behavior)
    # This will move CPU tensor to GPU
    result = st_gpu + st_cpu.cuda()
    assert result.is_cuda
    assert torch.allclose(result, torch.ones(5, 5, device="cuda") * 2)


@skip_cuda
def test_gpu_memory_sharing():
    """Test memory sharing on GPU."""
    # Create GPU tensor
    pt_gpu = torch.randn(10, 10, dtype=torch.float32, device="cuda")
    st_gpu = SupersayanTensor(pt_gpu)

    # Check that they share the same underlying storage
    pt_gpu[0, 0] = 999.0
    assert st_gpu[0, 0].item() == 999.0

    # Modifications to SupersayanTensor should affect original
    st_gpu[1, 1] = 888.0
    assert pt_gpu[1, 1].item() == 888.0


@skip_cuda
def test_gpu_constructors():
    """Test GPU tensor constructors."""
    device = "cuda"

    # Test zeros
    st_zeros = SupersayanTensor.zeros(5, 5, device=device)
    assert st_zeros.is_cuda
    assert torch.allclose(st_zeros, torch.zeros(5, 5, device=device))

    # Test ones
    st_ones = SupersayanTensor.ones(5, 5, device=device)
    assert st_ones.is_cuda
    assert torch.allclose(st_ones, torch.ones(5, 5, device=device))

    # Test randn
    st_randn = SupersayanTensor.randn(5, 5, device=device)
    assert st_randn.is_cuda
    assert st_randn.shape == (5, 5)


@skip_cuda
def test_dlpack_gpu():
    """Test DLPack interface on GPU."""
    # Create GPU tensor
    st_gpu = SupersayanTensor.randn(10, 10, device="cuda")

    # Export via DLPack
    dlpack = st_gpu.to_dlpack()

    # Import back
    st_gpu2 = SupersayanTensor.from_dlpack(dlpack)
    assert st_gpu2.is_cuda
    assert torch.allclose(st_gpu, st_gpu2)

    # Test with PyTorch
    pt_gpu = torch_dlpack.from_dlpack(st_gpu.to_dlpack())
    assert pt_gpu.is_cuda
    assert torch.allclose(st_gpu, pt_gpu)
