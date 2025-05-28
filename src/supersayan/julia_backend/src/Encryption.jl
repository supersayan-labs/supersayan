module Encryption

using Random, LinearAlgebra
using PythonCall
using CUDA

import SupersayanTFHE.Constants
import SupersayanTFHE.Types: LWE, LWE_ARRAY, A, B, MU, SIGMA, KEY, P, pack_lwe, extract_lwe

"""
Map a Float32 onto the signed torus (-0.5, 0.5]
"""
function real_to_torus(x::Float32)::Float32
    x = mod(x, 1.0)
    return x - (x >= 0.5 ? 1.0 : 0.0)
end

"""
Convert signed torus (-0.5,0.5] back to real [0,1)
"""
function torus_to_real(x::Float32)::Float32
    return x < 0 ? x + 1.0f0 : x
end

"""
Project a torus value to the discrete grid of size p
"""
function project_to_discrete_torus(x::Float32, p::P)::Float32
    return real_to_torus(round(x * p) / p)
end

"""
Encrypt a single message to an LWE ciphertext
"""
function encrypt_to_lwe(mu::MU, key::KEY, sigma::SIGMA)::LWE
    a = real_to_torus.(Float32.(rand(length(key))))
    e = sigma * randn(Float32)
    b = real_to_torus(mu + dot(key, a) + e)
    return pack_lwe(a, b)
end

"""
GPU kernel for encrypting multiple messages in parallel
"""
function encrypt_kernel!(
    ct::CuDeviceMatrix{Float32},
    mus::CuDeviceVector{Float32},
    key::CuDeviceVector{Float32},
    sigma::Float32,
    n_key::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= length(mus)
        # Generate random values for this ciphertext
        # Note: This is a simplified approach. In production, you'd want to use proper GPU random number generation
        Random.seed!(idx)

        # Compute a values
        a_sum = 0.0f0
        for j = 1:n_key
            a_val = real_to_torus(rand(Float32))
            ct[idx, j+1] = a_val
            a_sum += key[j] * a_val
        end

        # Compute b with noise
        e = sigma * randn(Float32)
        b = real_to_torus(mus[idx] + a_sum + e)
        ct[idx, 1] = b
    end

    return nothing
end

"""
CPU version of encrypt_to_lwes for regular arrays
"""
function encrypt_to_lwes_cpu(
    mus::AbstractArray{MU},
    key::KEY,
    sigma::SIGMA = Constants.sigma,
)::LWE_ARRAY
    n = length(key) + 1
    ct = Matrix{Float32}(undef, length(mus), n)

    for (i, mu) in enumerate(mus)
        ct[i, :] .= encrypt_to_lwe(mu, key, sigma)
    end

    return ct
end

"""
GPU version of encrypt_to_lwes for CuArrays
"""
function encrypt_to_lwes_gpu(
    mus::CuArray{MU},
    key::KEY,
    sigma::SIGMA = Constants.sigma,
)::CuArray{Float32,2}
    n_messages = length(mus)
    n_key = length(key)
    n = n_key + 1

    # Allocate output on GPU
    ct = CUDA.zeros(Float32, n_messages, n)

    # Convert key to GPU if needed
    key_gpu = CuArray(key)

    # For now, we'll use a simple approach: copy to CPU, encrypt, copy back
    # This avoids the scalar indexing issue but isn't optimal for performance
    mus_cpu = Array(mus)
    ct_cpu = encrypt_to_lwes_cpu(mus_cpu, key, sigma)
    copyto!(ct, ct_cpu)

    return ct
end

"""
Encrypt an array of messages to LWE ciphertexts
"""
function encrypt_to_lwes(
    mus::AbstractArray{MU},
    key::KEY,
    sigma::SIGMA = Constants.sigma,
)::Union{LWE_ARRAY,CuArray{Float32,2}}
    if isa(mus, CuArray)
        return encrypt_to_lwes_gpu(mus, key, sigma)
    else
        return encrypt_to_lwes_cpu(mus, key, sigma)
    end
end

"""
Decrypt a single LWE ciphertext, returning a Float32 in [0,1)
"""
function decrypt_from_lwe(c::LWE, key::KEY, p::P = Constants.p)::MU
    a, b = extract_lwe(c)
    phase = real_to_torus(b - dot(key, a))
    discrete = project_to_discrete_torus(phase, p)
    return torus_to_real(discrete)
end

"""
CPU version of decrypt_from_lwes
"""
function decrypt_from_lwes_cpu(
    ciphertexts::LWE_ARRAY,
    key::KEY,
    p::P = Constants.p,
)::AbstractArray{MU}
    return [
        decrypt_from_lwe(view(ciphertexts, i, :), key, p) for i = 1:size(ciphertexts, 1)
    ]
end

"""
GPU version of decrypt_from_lwes
"""
function decrypt_from_lwes_gpu(
    ciphertexts::CuArray{Float32,2},
    key::KEY,
    p::P = Constants.p,
)::CuArray{MU}
    # For now, copy to CPU, decrypt, and copy back
    # This avoids scalar indexing issues
    ciphertexts_cpu = Array(ciphertexts)
    decrypted_cpu = decrypt_from_lwes_cpu(ciphertexts_cpu, key, p)
    return CuArray(decrypted_cpu)
end

"""
Decrypt a batch of LWE ciphertexts, returning an array of Float32 in [0,1)
"""
function decrypt_from_lwes(
    ciphertexts::Union{LWE_ARRAY,CuArray{Float32,2}},
    key::KEY,
    p::P = Constants.p,
)::Union{AbstractArray{MU},CuArray{MU}}
    if isa(ciphertexts, CuArray)
        return decrypt_from_lwes_gpu(ciphertexts, key, p)
    else
        return decrypt_from_lwes_cpu(ciphertexts, key, p)
    end
end

"""
Generate a secret key for encryption/decryption
"""
function generate_key()::KEY
    return rand(Constants.S, Constants.n)
end

export encrypt_to_lwes, decrypt_from_lwes, generate_key, torus_to_real

end
