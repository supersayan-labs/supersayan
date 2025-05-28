import numpy as np
import pytest

from supersayan.core.bindings import SupersayanTFHE
from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.core.types import SupersayanTensor

np.random.seed(123)

EPSILON = 0.1


def _mod1(x: np.ndarray) -> np.ndarray:
    """Map values onto the unit interval [0, 1)."""
    return np.mod(x, 1.0)


def _assert_close(arr1: np.ndarray, arr2: np.ndarray, eps: float = EPSILON):
    """Assert that two arrays are element-wise within *eps* (max absolute error)."""
    max_delta = np.max(np.abs(arr1.to_numpy() - arr2.to_numpy()))
    assert max_delta <= eps, f"Maximum delta {max_delta} exceeds epsilon {eps}"


@pytest.fixture(scope="module")
def secret_key():
    """Generate a secret key once for all operation tests."""
    return generate_secret_key()


def test_add_lwe_ciphertext_ciphertext(secret_key):
    """Test homomorphic addition of two ciphertext vectors (LWE_ARRAY + LWE_ARRAY)."""
    lhs_plain = SupersayanTensor(
        np.random.uniform(0.0, 0.5, size=20).astype(np.float32)
    )
    rhs_plain = SupersayanTensor(
        np.random.uniform(0.0, 0.5, size=20).astype(np.float32)
    )

    lhs_ct = encrypt_to_lwes(lhs_plain, secret_key)
    rhs_ct = encrypt_to_lwes(rhs_plain, secret_key)

    res_ct = SupersayanTFHE.Operations.add_lwe(lhs_ct.to_julia(), rhs_ct.to_julia())

    decrypted = decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), secret_key)
    expected = _mod1(lhs_plain + rhs_plain)

    _assert_close(decrypted, expected)


@pytest.mark.parametrize("scalar", [0.1, 0.25, 0.4])
def test_add_lwe_ciphertext_scalar(secret_key, scalar):
    """Test homomorphic addition of a ciphertext vector with a scalar (LWE_ARRAY + Float32)."""
    plain = SupersayanTensor(np.random.uniform(0.0, 0.5, size=30).astype(np.float32))
    ct = encrypt_to_lwes(plain, secret_key)

    res_ct = SupersayanTFHE.Operations.add_lwe(ct.to_julia(), np.float32(scalar))

    decrypted = decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), secret_key)
    expected = _mod1(plain + scalar)

    _assert_close(decrypted, expected)


@pytest.mark.parametrize("scalar", [0.05, 0.15, 0.3])
def test_add_lwe_scalar_ciphertext(secret_key, scalar):
    """Test commutative form where the scalar is the first argument (Float32 + LWE_ARRAY)."""
    plain = SupersayanTensor(np.random.uniform(0.0, 0.5, size=25).astype(np.float32))
    ct = encrypt_to_lwes(plain, secret_key)

    res_ct = SupersayanTFHE.Operations.add_lwe(np.float32(scalar), ct.to_julia())

    decrypted = decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), secret_key)
    expected = _mod1(plain + scalar)

    _assert_close(decrypted, expected)


def test_add_lwe_large_vector_parallel(secret_key):
    """Test vector size > 100 to exercise the multi-threaded implementation path."""
    lhs_plain = SupersayanTensor(
        np.random.uniform(0.0, 0.5, size=150).astype(np.float32)
    )
    rhs_plain = SupersayanTensor(
        np.random.uniform(0.0, 0.5, size=150).astype(np.float32)
    )

    lhs_ct = encrypt_to_lwes(lhs_plain, secret_key)
    rhs_ct = encrypt_to_lwes(rhs_plain, secret_key)

    res_ct = SupersayanTFHE.Operations.add_lwe(lhs_ct.to_julia(), rhs_ct.to_julia())

    decrypted = decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), secret_key)
    expected = _mod1(lhs_plain + rhs_plain)

    _assert_close(decrypted, expected)
