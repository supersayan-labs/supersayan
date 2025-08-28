module Operations

using Base.Threads
using LinearAlgebra: BLAS
using CUDA

import ..Types: LWE, LWE_ARRAY, LWE_BATCH, pack_lwe, extract_lwe

# Match BLAS threads to Julia threads for maximum throughput
BLAS.set_num_threads(Threads.nthreads())

# Helper validation functions
_validate_scalar(x::Float32) = @assert isfinite(x) "Scalar must be finite: got $x"
_validate_same_shape(a, b) =
    @assert size(a) == size(b) "Shapes must match: got $(size(a)) and $(size(b))"
_validate_lwe_array(x) =
    @assert ndims(x) == 2 && !isempty(x) "Expected non-empty 2D matrix, got $(ndims(x))D array"

"""
Homomorphic addition of two LWE ciphertexts.
"""
function add_lwe(lhs::LWE, rhs::LWE)::LWE
    _validate_same_shape(lhs, rhs)
    a_1, b_1 = extract_lwe(lhs)
    a_2, b_2 = extract_lwe(rhs)
    return pack_lwe(a_1 .+ a_2, b_1 + b_2)
end

"""
Add a plaintext scalar to an LWE ciphertext.
"""
function add_lwe(lhs::LWE, rhs::Float32)::LWE
    _validate_scalar(rhs)
    a, b = extract_lwe(lhs)
    return pack_lwe(a, b + rhs)
end

"""
Commutative form.
"""
add_lwe(lhs::Float32, rhs::LWE)::LWE = add_lwe(rhs, lhs)

"""
GPU kernel for adding two LWE arrays
"""
function add_lwe_kernel!(
    out::CuDeviceMatrix{Float32},
    lhs::CuDeviceMatrix{Float32},
    rhs::CuDeviceMatrix{Float32},
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if idx <= size(out, 1) && idy <= size(out, 2)
        out[idx, idy] = lhs[idx, idy] + rhs[idx, idy]
    end

    return nothing
end

"""
GPU kernel for adding scalar to LWE array
"""
function add_lwe_scalar_kernel!(
    out::CuDeviceMatrix{Float32},
    lhs::CuDeviceMatrix{Float32},
    rhs::Float32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if idx <= size(out, 1) && idy <= size(out, 2)
        # Only add scalar to b component (first column)
        if idy == 1
            out[idx, idy] = lhs[idx, idy] + rhs
        else
            out[idx, idy] = lhs[idx, idy]
        end
    end

    return nothing
end

"""
Add two arrays of LWE ciphertexts.
"""
function add_lwe(lhs::LWE_ARRAY, rhs::LWE_ARRAY)::LWE_ARRAY
    _validate_same_shape(lhs, rhs)
    _validate_lwe_array(lhs)
    out = Matrix{Float32}(undef, size(lhs))
    return _vector_op!(add_lwe, out, lhs, rhs)
end

"""
Add two GPU arrays of LWE ciphertexts.
"""
function add_lwe(lhs::CuArray{Float32,2}, rhs::CuArray{Float32,2})::CuArray{Float32,2}
    _validate_same_shape(lhs, rhs)
    out = CUDA.zeros(Float32, size(lhs))

    threads = (16, 16)
    blocks = (cld(size(out, 1), threads[1]), cld(size(out, 2), threads[2]))

    @cuda threads=threads blocks=blocks add_lwe_kernel!(out, lhs, rhs)

    return out
end

"""
Add an array of LWE ciphertexts to a plaintext scalar.
"""
function add_lwe(lhs::LWE_ARRAY, rhs::Float32)::LWE_ARRAY
    _validate_lwe_array(lhs)
    _validate_scalar(rhs)
    out = Matrix{Float32}(undef, size(lhs))
    rhs = fill(rhs, size(lhs, 1))
    return _vector_op!(add_lwe, out, lhs, rhs)
end

"""
Add a GPU array of LWE ciphertexts to a plaintext scalar.
"""
function add_lwe(lhs::CuArray{Float32,2}, rhs::Float32)::CuArray{Float32,2}
    _validate_scalar(rhs)
    out = CUDA.zeros(Float32, size(lhs))

    threads = (16, 16)
    blocks = (cld(size(out, 1), threads[1]), cld(size(out, 2), threads[2]))

    @cuda threads=threads blocks=blocks add_lwe_scalar_kernel!(out, lhs, rhs)

    return out
end

"""
Commutative form.
"""
add_lwe(lhs::Float32, rhs::LWE_ARRAY)::LWE_ARRAY = add_lwe(rhs, lhs)
add_lwe(lhs::Float32, rhs::CuArray{Float32,2})::CuArray{Float32,2} = add_lwe(rhs, lhs)

"""
Apply a function element-wise to two arrays.
"""
function _vector_op!(f, out::AbstractMatrix, a::AbstractMatrix, b::AbstractArray)
    @assert size(out) == size(a) &&
            size(a, 1) == (b isa AbstractMatrix ? size(b, 1) : length(b)) "Dimension mismatch"

    n = size(out, 1)

    if n > 100
        @threads for i = 1:n
            lhs_row = @view a[i, :]
            rhs_arg = b isa AbstractMatrix ? (@view b[i, :]) : b[i]
            out[i, :] = f(lhs_row, rhs_arg)
        end
    else
        for i = 1:n
            lhs_row = @view a[i, :]
            rhs_arg = b isa AbstractMatrix ? (@view b[i, :]) : b[i]
            out[i, :] = f(lhs_row, rhs_arg)
        end
    end
    return out
end

"""
GPU kernel for multiplying LWE by scalar
"""
function mult_lwe_scalar_kernel!(
    out::CuDeviceMatrix{Float32},
    lhs::CuDeviceMatrix{Float32},
    rhs::Float32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if idx <= size(out, 1) && idy <= size(out, 2)
        out[idx, idy] = lhs[idx, idy] * rhs
    end

    return nothing
end

"""
Multiply a plaintext scalar to an LWE ciphertext.
"""
function mult_lwe(lhs::LWE, rhs::Float32)::LWE
    _validate_scalar(rhs)
    a, b = extract_lwe(lhs)
    return pack_lwe(a .* rhs, b * rhs)
end

"""
Commutative form.
"""
mult_lwe(lhs::Float32, rhs::LWE)::LWE = mult_lwe(rhs, lhs)

"""
Multiply an array of LWE ciphertexts to a plaintext scalar.
"""
function mult_lwe(lhs::LWE_ARRAY, rhs::Float32)::LWE_ARRAY
    _validate_lwe_array(lhs)
    _validate_scalar(rhs)
    out = Matrix{Float32}(undef, size(lhs))
    rhs = fill(rhs, size(lhs, 1))
    return _vector_op!(mult_lwe, out, lhs, rhs)
end

"""
Multiply a GPU array of LWE ciphertexts to a plaintext scalar.
"""
function mult_lwe(lhs::CuArray{Float32,2}, rhs::Float32)::CuArray{Float32,2}
    _validate_scalar(rhs)
    out = CUDA.zeros(Float32, size(lhs))

    threads = (16, 16)
    blocks = (cld(size(out, 1), threads[1]), cld(size(out, 2), threads[2]))

    @cuda threads=threads blocks=blocks mult_lwe_scalar_kernel!(out, lhs, rhs)

    return out
end

"""
Commutative form.
"""
mult_lwe(lhs::Float32, rhs::LWE_ARRAY)::LWE_ARRAY = mult_lwe(rhs, lhs)
mult_lwe(lhs::Float32, rhs::CuArray{Float32,2})::CuArray{Float32,2} = mult_lwe(rhs, lhs)

"""
Compute the dot product of an array of LWE ciphertexts and an array of plaintext scalars. 
The `zero_cipher` argument provides a ciphertext with the correct shape to initialize the accumulator.
"""
function dot_product_lwe(
    enc::LWE_ARRAY,
    plain::AbstractArray{Float32},
    zero_cipher::LWE,
)::LWE
    _validate_lwe_array(enc)
    @assert ndims(plain) == 1 && size(enc, 1) == length(plain) "Expected 1D vector of length $(size(enc, 1)), got $(ndims(plain))D array of length $(length(plain))"
    @assert size(enc, 2) == length(zero_cipher) "LWE dimension mismatch: $(size(enc, 2)) vs $(length(zero_cipher))"
    @assert all(isfinite, plain) "All plaintext values must be finite"

    accumulator = copy(zero_cipher)

    for i = 1:size(enc, 1)
        enc_i = @view enc[i, :]

        product = mult_lwe(enc_i, plain[i])

        accumulator = add_lwe(accumulator, product)
    end

    return accumulator
end

"""
Compute batch dot products of LWE ciphertexts and plaintext scalars.
"""
function batch_dot_product_lwe(
    enc_batch::LWE_BATCH,
    plain_batch::AbstractMatrix{Float32},
    zero_cipher::LWE,
)::LWE_ARRAY
    @assert ndims(enc_batch) == 3 && ndims(plain_batch) == 2 "Expected 3D and 2D arrays"

    batch_size, feature_dim, lwe_dim = size(enc_batch)
    @assert size(plain_batch) == (batch_size, feature_dim) && length(zero_cipher) == lwe_dim "Dimension mismatch"
    @assert all(isfinite, plain_batch) "All plaintext values must be finite"

    out = Matrix{Float32}(undef, batch_size, lwe_dim)

    for b = 1:batch_size
        enc_b = @view enc_batch[b, :, :]
        plain_b = @view plain_batch[b, :]

        result = dot_product_lwe(enc_b, plain_b, zero_cipher)
        out[b, :] = result
    end

    return out
end

end
