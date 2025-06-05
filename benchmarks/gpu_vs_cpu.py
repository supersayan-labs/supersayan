import numpy as np
import pytest
import torch

from supersayan.core.bindings import SupersayanTFHE, HAS_CUPY, cp
from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.core.types import SupersayanTensor
from supersayan.nn.layers.linear import Linear
from supersayan.logging_config import configure_logging, get_logger

configure_logging(level="INFO")
logger = get_logger(__name__)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
skip_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

np.random.seed(42)
if HAS_CUPY:
    cp.random.seed(42)

EPSILON = 0.1


def _mod1(x: np.ndarray) -> np.ndarray:
    """Map values to the unit interval [0, 1)."""
    return np.mod(x, 1.0)


@pytest.fixture(scope="module")
def fhe_secret_key():
    """Generate secret key fixture for FHE operations."""
    logger.info("Generating secret key (fixture)...")
    return generate_secret_key()


# Fixtures for encryption/decryption tests (support multi-dimensional arrays)
@pytest.fixture(params=[
    (1000, 1000, 1000),     # 3D medium
])
def encryption_shape_fixture(request):
    """Parameterized fixture for encryption/decryption tensor shapes."""
    return request.param


# Fixtures for homomorphic operations (1D arrays only)
@pytest.fixture(params=[
    (20,),           # Small vector
    (100,),          # Medium vector  
    (500,),          # Large vector
    (1000,),         # Very large vector
])
def operation_shape_fixture(request):
    """Parameterized fixture for homomorphic operation shapes (1D only)."""
    return request.param


@pytest.fixture(params=[
    (8, 4),          # Linear small
    (64, 32),        # Linear medium
    (256, 128),      # Linear large
])
def linear_shape_fixture(request):
    """Parameterized fixture for linear layer shapes (in_features, out_features)."""
    return request.param


@pytest.fixture(params=[
    (5, 10),         # Batch small
    (16, 64),        # Batch medium
    (32, 256),       # Batch large
])
def batch_shape_fixture(request):
    """Parameterized fixture for batch operations (batch_size, feature_dim)."""
    return request.param


def create_cpu_tensor(shape):
    """Create a CPU tensor with given shape."""
    return SupersayanTensor(np.random.uniform(0.0, 0.5, size=shape).astype(np.float32))


def create_gpu_tensor(shape):
    """Create a GPU tensor with given shape."""
    if not cuda_available or not HAS_CUPY:
        pytest.skip("CUDA or CuPy not available")
    return SupersayanTensor(
        cp.random.uniform(0.0, 0.5, size=shape).astype(np.float32),
        device=torch.device("cuda")
    )


# ============================================================================
# ENCRYPTION/DECRYPTION BENCHMARKS
# ============================================================================

def test_encryption_cpu_benchmark(benchmark, fhe_secret_key, encryption_shape_fixture):
    """Benchmark CPU encryption performance."""
    tensor = create_cpu_tensor(encryption_shape_fixture)
    
    def encrypt_decrypt():
        encrypted = encrypt_to_lwes(tensor, fhe_secret_key)
        decrypted = decrypt_from_lwes(encrypted, fhe_secret_key)
        return decrypted
    
    result = benchmark(encrypt_decrypt)
    assert result.shape == tensor.shape


@skip_cuda
def test_encryption_gpu_benchmark(benchmark, fhe_secret_key, encryption_shape_fixture):
    """Benchmark GPU encryption performance."""
    tensor = create_gpu_tensor(encryption_shape_fixture)
    
    def encrypt_decrypt():
        encrypted = encrypt_to_lwes(tensor, fhe_secret_key)
        decrypted = decrypt_from_lwes(encrypted, fhe_secret_key)
        return decrypted
    
    result = benchmark(encrypt_decrypt)
    assert result.shape == tensor.shape


# ============================================================================
# ADDITION OPERATION BENCHMARKS (1D arrays only)
# ============================================================================

def test_add_ciphertext_ciphertext_cpu_benchmark(benchmark, fhe_secret_key, operation_shape_fixture):
    """Benchmark CPU homomorphic addition of two ciphertexts."""
    lhs = create_cpu_tensor(operation_shape_fixture)
    rhs = create_cpu_tensor(operation_shape_fixture)
    
    def add_operation():
        lhs_ct = encrypt_to_lwes(lhs, fhe_secret_key)
        rhs_ct = encrypt_to_lwes(rhs, fhe_secret_key)
        res_ct = SupersayanTFHE.Operations.add_lwe(lhs_ct.to_julia(), rhs_ct.to_julia())
        return decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), fhe_secret_key)
    
    result = benchmark(add_operation)
    assert result.shape == lhs.shape


@skip_cuda
def test_add_ciphertext_ciphertext_gpu_benchmark(benchmark, fhe_secret_key, operation_shape_fixture):
    """Benchmark GPU homomorphic addition of two ciphertexts."""
    lhs = create_gpu_tensor(operation_shape_fixture)
    rhs = create_gpu_tensor(operation_shape_fixture)
    
    def add_operation():
        lhs_ct = encrypt_to_lwes(lhs, fhe_secret_key)
        rhs_ct = encrypt_to_lwes(rhs, fhe_secret_key)
        res_ct = SupersayanTFHE.Operations.add_lwe(lhs_ct.to_julia(), rhs_ct.to_julia())
        return decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), fhe_secret_key)
    
    result = benchmark(add_operation)
    assert result.shape == lhs.shape


def test_add_ciphertext_scalar_cpu_benchmark(benchmark, fhe_secret_key, operation_shape_fixture):
    """Benchmark CPU homomorphic addition of ciphertext and scalar."""
    tensor = create_cpu_tensor(operation_shape_fixture)
    scalar = np.float32(0.25)
    
    def add_scalar_operation():
        ct = encrypt_to_lwes(tensor, fhe_secret_key)
        res_ct = SupersayanTFHE.Operations.add_lwe(ct.to_julia(), scalar)
        return decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), fhe_secret_key)
    
    result = benchmark(add_scalar_operation)
    assert result.shape == tensor.shape


@skip_cuda
def test_add_ciphertext_scalar_gpu_benchmark(benchmark, fhe_secret_key, operation_shape_fixture):
    """Benchmark GPU homomorphic addition of ciphertext and scalar."""
    tensor = create_gpu_tensor(operation_shape_fixture)
    scalar = np.float32(0.25)
    
    def add_scalar_operation():
        ct = encrypt_to_lwes(tensor, fhe_secret_key)
        res_ct = SupersayanTFHE.Operations.add_lwe(ct.to_julia(), scalar)
        return decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), fhe_secret_key)
    
    result = benchmark(add_scalar_operation)
    assert result.shape == tensor.shape


# ============================================================================
# MULTIPLICATION OPERATION BENCHMARKS (1D arrays only)
# ============================================================================

def test_mult_ciphertext_scalar_cpu_benchmark(benchmark, fhe_secret_key, operation_shape_fixture):
    """Benchmark CPU homomorphic multiplication of ciphertext and scalar."""
    tensor = create_cpu_tensor(operation_shape_fixture)
    scalar = np.float32(0.3)
    
    def mult_scalar_operation():
        ct = encrypt_to_lwes(tensor, fhe_secret_key)
        res_ct = SupersayanTFHE.Operations.mult_lwe(ct.to_julia(), scalar)
        return decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), fhe_secret_key)
    
    result = benchmark(mult_scalar_operation)
    assert result.shape == tensor.shape


@skip_cuda
def test_mult_ciphertext_scalar_gpu_benchmark(benchmark, fhe_secret_key, operation_shape_fixture):
    """Benchmark GPU homomorphic multiplication of ciphertext and scalar."""
    tensor = create_gpu_tensor(operation_shape_fixture)
    scalar = np.float32(0.3)
    
    def mult_scalar_operation():
        ct = encrypt_to_lwes(tensor, fhe_secret_key)
        res_ct = SupersayanTFHE.Operations.mult_lwe(ct.to_julia(), scalar)
        return decrypt_from_lwes(SupersayanTensor._from_julia(res_ct), fhe_secret_key)
    
    result = benchmark(mult_scalar_operation)
    assert result.shape == tensor.shape

# ============================================================================
# LINEAR LAYER BENCHMARKS
# ============================================================================

def test_linear_layer_cpu_benchmark(benchmark, fhe_secret_key, linear_shape_fixture):
    """Benchmark CPU Linear layer forward pass."""
    in_features, out_features = linear_shape_fixture
    
    linear = Linear(in_features=in_features, out_features=out_features, bias=True)
    input_data = SupersayanTensor(torch.rand(1, in_features))
    
    def linear_forward():
        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)
        output_encrypted = linear(encrypted_input)
        return decrypt_from_lwes(output_encrypted, fhe_secret_key)
    
    result = benchmark(linear_forward)
    assert result.shape == (1, out_features)


@skip_cuda
def test_linear_layer_gpu_benchmark(benchmark, fhe_secret_key, linear_shape_fixture):
    """Benchmark GPU Linear layer forward pass."""
    in_features, out_features = linear_shape_fixture
    
    linear = Linear(in_features=in_features, out_features=out_features, bias=True)
    input_data = SupersayanTensor(torch.rand(1, in_features), device=torch.device("cuda"))
    
    def linear_forward():
        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)
        output_encrypted = linear(encrypted_input)
        return decrypt_from_lwes(output_encrypted, fhe_secret_key)
    
    result = benchmark(linear_forward)
    assert result.shape == (1, out_features)


# ============================================================================
# COMBINED OPERATION BENCHMARKS
# ============================================================================

def test_combined_operations_cpu_benchmark(benchmark, fhe_secret_key):
    """Benchmark a combination of operations on CPU."""
    vec_size = 64
    
    def combined_operations():
        # Create input tensors (1D only for homomorphic operations)
        tensor1 = SupersayanTensor(
            np.random.uniform(0.0, 0.3, size=vec_size).astype(np.float32)
        )
        tensor2 = SupersayanTensor(
            np.random.uniform(0.0, 0.3, size=vec_size).astype(np.float32)
        )
        scalar = np.float32(0.2)
        
        # Encrypt
        ct1 = encrypt_to_lwes(tensor1, fhe_secret_key)
        ct2 = encrypt_to_lwes(tensor2, fhe_secret_key)
        
        # Add two ciphertexts
        add_ct = SupersayanTFHE.Operations.add_lwe(ct1.to_julia(), ct2.to_julia())
        add_result = SupersayanTensor._from_julia(add_ct)
        
        # Multiply result by scalar
        mult_ct = SupersayanTFHE.Operations.mult_lwe(add_result.to_julia(), scalar)
        mult_result = SupersayanTensor._from_julia(mult_ct)
        
        # Decrypt final result
        return decrypt_from_lwes(mult_result, fhe_secret_key)
    
    result = benchmark(combined_operations)
    assert result.shape == (vec_size,)


@skip_cuda
def test_combined_operations_gpu_benchmark(benchmark, fhe_secret_key):
    """Benchmark a combination of operations on GPU."""
    vec_size = 64
    
    def combined_operations():
        # Create input tensors (1D only for homomorphic operations)
        tensor1 = SupersayanTensor(
            cp.random.uniform(0.0, 0.3, size=vec_size).astype(np.float32),
            device=torch.device("cuda")
        )
        tensor2 = SupersayanTensor(
            cp.random.uniform(0.0, 0.3, size=vec_size).astype(np.float32),
            device=torch.device("cuda")
        )
        scalar = np.float32(0.2)
        
        # Encrypt
        ct1 = encrypt_to_lwes(tensor1, fhe_secret_key)
        ct2 = encrypt_to_lwes(tensor2, fhe_secret_key)
        
        # Add two ciphertexts
        add_ct = SupersayanTFHE.Operations.add_lwe(ct1.to_julia(), ct2.to_julia())
        add_result = SupersayanTensor._from_julia(add_ct)
        
        # Multiply result by scalar
        mult_ct = SupersayanTFHE.Operations.mult_lwe(add_result.to_julia(), scalar)
        mult_result = SupersayanTensor._from_julia(mult_ct)
        
        # Decrypt final result
        return decrypt_from_lwes(mult_result, fhe_secret_key)
    
    result = benchmark(combined_operations)
    assert result.shape == (vec_size,)
