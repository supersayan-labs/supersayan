import numpy as np
import pytest

from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt_to_lwes, decrypt_from_lwes
from supersayan.core.bindings import SupersayanTFHE

# Set fixed seed for reproducibility
np.random.seed(456)

EPSILON = 0.1


def _mod1(x: np.ndarray) -> np.ndarray:
    """Map values to the unit interval [0, 1)."""
    return np.mod(x, 1.0).astype(np.float32)


def _assert_close(arr1: np.ndarray, arr2: np.ndarray, eps: float = EPSILON):
    """Assert that two arrays are element-wise within *eps*."""
    max_delta = np.max(np.abs(arr1 - arr2))
    assert max_delta <= eps, f"Maximum delta {max_delta} exceeds epsilon {eps}"


@pytest.fixture(scope="module")
def secret_key():
    """Generate a secret key once for all multiplication/dot-product tests."""
    return generate_secret_key()


# -----------------------------------------------------------------------------
# Scalar multiplication tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("scalar", [0.1, 0.3, 0.45])
def test_mult_ciphertext_scalar(secret_key, scalar):
    """Homomorphic multiplication of ciphertext vector by a scalar (LWE_ARRAY × Float32)."""
    plain = np.random.uniform(0.0, 0.5, size=32).astype(np.float32)
    ct = encrypt_to_lwes(plain, secret_key)

    res_ct = SupersayanTFHE.Operations.mult_lwe(ct, np.float32(scalar))

    decrypted = decrypt_from_lwes(res_ct, secret_key)
    expected = _mod1(plain * scalar)

    _assert_close(decrypted, expected)


@pytest.mark.parametrize("scalar", [0.05, 0.2, 0.35])
def test_mult_scalar_ciphertext(secret_key, scalar):
    """Commutative form where scalar is the first argument (Float32 × LWE_ARRAY)."""
    plain = np.random.uniform(0.0, 0.5, size=40).astype(np.float32)
    ct = encrypt_to_lwes(plain, secret_key)

    res_ct = SupersayanTFHE.Operations.mult_lwe(np.float32(scalar), ct)

    decrypted = decrypt_from_lwes(res_ct, secret_key)
    expected = _mod1(plain * scalar)

    _assert_close(decrypted, expected)


def test_mult_large_vector_parallel(secret_key):
    """Vector length > 100 to exercise multithreaded path in `mult`."""
    plain = np.random.uniform(0.0, 0.5, size=150).astype(np.float32)
    scalar = np.float32(0.37)
    ct = encrypt_to_lwes(plain, secret_key)

    res_ct = SupersayanTFHE.Operations.mult_lwe(ct, scalar)

    decrypted = decrypt_from_lwes(res_ct, secret_key)
    expected = _mod1(plain * scalar)

    _assert_close(decrypted, expected)


# -----------------------------------------------------------------------------
# Dot-product tests
# -----------------------------------------------------------------------------


def test_dot_product(secret_key):
    """Encrypted vector x plaintext vector dot-product."""
    vec_len = 20
    enc_plain = np.random.uniform(0.0, 0.5, size=vec_len).astype(np.float32)
    plain_weights = np.random.uniform(-0.5, 0.5, size=vec_len).astype(np.float32)

    enc_ct = encrypt_to_lwes(enc_plain, secret_key)

    zero_ct = encrypt_to_lwes(np.array([0.0], dtype=np.float32), secret_key)[0]

    res_ct = SupersayanTFHE.Operations.dot_product_lwe(enc_ct, plain_weights, zero_ct)

    decrypted = decrypt_from_lwes(np.expand_dims(res_ct, 0), secret_key)[0]
    expected = _mod1(np.dot(enc_plain, plain_weights))

    _assert_close(np.array([decrypted]), np.array([expected]))


def test_batch_dot_product(secret_key):
    """Test batch encrypted vector x plaintext vector dot-products."""
    batch_size = 5
    feature_dim = 10

    # Create batch of plaintext vectors to encrypt
    enc_plain_batch = np.random.uniform(
        0.0, 0.3, size=(batch_size, feature_dim)
    ).astype(np.float32)

    # Create batch of plaintext weight vectors
    plain_weights_batch = np.random.uniform(
        -0.5, 0.5, size=(batch_size, feature_dim)
    ).astype(np.float32)

    # Encrypt each vector in the batch
    enc_ct_batch = []
    for i in range(batch_size):
        enc_ct_batch.append(encrypt_to_lwes(enc_plain_batch[i], secret_key))

    # Stack into 3D array (batch_size, feature_dim, lwe_dim)
    enc_ct_batch = np.stack(enc_ct_batch, axis=0)

    # Get zero ciphertext for initialization
    zero_ct = encrypt_to_lwes(np.array([0.0], dtype=np.float32), secret_key)[0]

    # Perform batch dot product
    res_ct_batch = SupersayanTFHE.Operations.batch_dot_product_lwe(
        enc_ct_batch, plain_weights_batch, zero_ct
    )

    # Decrypt results
    decrypted_batch = decrypt_from_lwes(res_ct_batch, secret_key)

    # Compute expected results
    expected_batch = np.array(
        [
            _mod1(np.dot(enc_plain_batch[i], plain_weights_batch[i]))
            for i in range(batch_size)
        ]
    )

    _assert_close(decrypted_batch, expected_batch)


@pytest.mark.parametrize("batch_size,feature_dim", [(3, 8), (15, 20), (1, 50)])
def test_batch_dot_product_various_sizes(secret_key, batch_size, feature_dim):
    """Test batch dot product with various batch and feature dimensions."""
    # Create batch of plaintext vectors to encrypt
    enc_plain_batch = np.random.uniform(
        0.0, 0.2, size=(batch_size, feature_dim)
    ).astype(np.float32)

    # Create batch of plaintext weight vectors
    plain_weights_batch = np.random.uniform(
        -0.3, 0.3, size=(batch_size, feature_dim)
    ).astype(np.float32)

    # Encrypt each vector in the batch
    enc_ct_batch = []
    for i in range(batch_size):
        enc_ct_batch.append(encrypt_to_lwes(enc_plain_batch[i], secret_key))

    # Stack into 3D array (batch_size, feature_dim, lwe_dim)
    enc_ct_batch = np.stack(enc_ct_batch, axis=0)

    # Get zero ciphertext for initialization
    zero_ct = encrypt_to_lwes(np.array([0.0], dtype=np.float32), secret_key)[0]

    # Perform batch dot product
    res_ct_batch = SupersayanTFHE.Operations.batch_dot_product_lwe(
        enc_ct_batch, plain_weights_batch, zero_ct
    )

    # Decrypt results
    decrypted_batch = decrypt_from_lwes(res_ct_batch, secret_key)

    # Compute expected results
    expected_batch = np.array(
        [
            _mod1(np.dot(enc_plain_batch[i], plain_weights_batch[i]))
            for i in range(batch_size)
        ]
    )

    _assert_close(decrypted_batch, expected_batch)
