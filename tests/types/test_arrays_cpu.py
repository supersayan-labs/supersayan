import numpy as np
import torch

from supersayan.core.bindings import jl
from supersayan.core.types import SupersayanTensor


def test_numpy_interop_cpu():
    """Test NumPy array interoperability on CPU."""
    # Create from NumPy
    np_arr = np.linspace(0, 1, 11, dtype=np.float32)
    st = SupersayanTensor(np_arr)

    # Test conversion back to NumPy
    np_back = st.to_numpy()
    assert np.allclose(np_arr, np_back)
    assert np.shares_memory(np_arr, np_back), "Should share memory with original array"

    # Test that operations preserve type
    st2 = st + 2.0
    assert isinstance(st2, SupersayanTensor)
    assert np.allclose(st2.to_numpy(), np_arr + 2.0)


def test_pytorch_interop_cpu():
    """Test PyTorch tensor interoperability on CPU."""
    # Create from PyTorch
    pt_tensor = torch.linspace(0, 1, 11, dtype=torch.float32)
    st = SupersayanTensor(pt_tensor)

    # Test conversion and memory sharing
    assert torch.allclose(pt_tensor, torch.from_numpy(st.to_numpy()))
    assert np.shares_memory(
        pt_tensor.numpy(), st.to_numpy()
    ), "Should share memory with PyTorch tensor"

    # Test that operations preserve type
    st2 = st + 2.0
    assert isinstance(st2, SupersayanTensor)
    assert torch.allclose(torch.from_numpy(st2.to_numpy()), pt_tensor + 2.0)


def test_mixed_operations():
    """Test operations between arrays from different backends."""
    # Create arrays from different sources
    np_arr = np.ones((5, 5), dtype=np.float32)
    pt_tensor = torch.ones((5, 5), dtype=torch.float32) * 2

    st_np = SupersayanTensor(np_arr)
    st_pt = SupersayanTensor(pt_tensor)

    # Test operations between them
    result = st_np + st_pt
    assert isinstance(result, SupersayanTensor)
    assert np.allclose(result.to_numpy(), np.ones((5, 5)) * 3)


def test_dtype_preservation():
    """Test that data types are preserved across conversions."""
    dtypes = [np.float32]

    for dtype in dtypes:
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        st = SupersayanTensor(np_arr)
        np_back = st.to_numpy()

        assert np_back.dtype == dtype, f"dtype {dtype} should be preserved"
        assert np.array_equal(np_arr, np_back)
