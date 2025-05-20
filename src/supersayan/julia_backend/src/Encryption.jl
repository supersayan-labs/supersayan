module Encryption

using Random, LinearAlgebra
using PythonCall

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
Encrypt an array of messages to LWE ciphertexts
"""
function encrypt_to_lwes(mus::AbstractArray{MU}, key::KEY, sigma::SIGMA = Constants.sigma)::LWE_ARRAY
    n  = length(key) + 1
    ct = Matrix{Float32}(undef, length(mus), n)
    
    for (i, mu) in enumerate(mus)
        ct[i, :] .= encrypt_to_lwe(mu, key, sigma)
    end

    return ct
end

"""
Decrypt a single LWE ciphertext, returning a Float32 in [0,1)
"""
function decrypt_from_lwe(c::LWE, key::KEY, p::P=Constants.p)::MU
    a, b = extract_lwe(c)
    phase = real_to_torus(b - dot(key, a))
    discrete = project_to_discrete_torus(phase, p)
    return torus_to_real(discrete)
end

"""
Decrypt a batch of LWE ciphertexts, returning an array of Float32 in [0,1)
"""
function decrypt_from_lwes(ciphertexts::LWE_ARRAY, key::KEY, p::P=Constants.p)::AbstractArray{MU}
    return [decrypt_from_lwe(view(ciphertexts, i, :), key, p) for i in 1:size(ciphertexts, 1)]
end

"""
Generate a secret key for encryption/decryption
"""
function generate_key()::KEY
    return rand(Constants.S, Constants.n)
end

export encrypt_to_lwes, decrypt_from_lwes, generate_key, torus_to_real

end
