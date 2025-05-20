module Encryption

using Random, LinearAlgebra
using PythonCall

import SupersayanTFHE.Constants
import SupersayanTFHE.Types: LWE, A, B, MU, SIGMA, KEY, P, pack_lwe, extract_lwe


function real_to_torus(x::Float32)::Float32
    x = mod(x, 1.0)
    return x - (x >= 0.5 ? 1.0 : 0.0)
end

function project_to_discrete_torus(x::Float32, p::P)::Float32
    return real_to_torus(round(x * p) / p)
end

function encrypt_to_lwe(mu::MU, key::KEY, sigma::SIGMA)::LWE
    a = real_to_torus.(Float32.(rand(length(key))))
    e = sigma * randn(Float32)
    b = real_to_torus(mu + dot(key, a) + e)
    return pack_lwe(a, b)
end


function encrypt_to_lwes(mus::AbstractArray{MU}, key::KEY, sigma::SIGMA=Constants.sigma)::AbstractArray{LWE}
    return [encrypt_to_lwe(m, key, sigma) for m in mus]
end

function decrypt_from_lwe(c::LWE, key::KEY, p::P)::MU
    a, b = extract_lwe(c)
    phase = real_to_torus(b - dot(key, a))
    return project_to_discrete_torus(phase, p)
end

function decrypt_from_lwes(ciphertexts::AbstractMatrix{Float32},
                            key::KEY, p::P = Constants.p)
    return [decrypt_from_lwe(view(ciphertexts, i, :), key, p)
            for i in 1:size(ciphertexts, 1)]
    # ou : map(row -> decrypt_from_lwe(row, key, p), eachrow(ciphertexts))
end

function generate_key()::KEY
    return rand(Constants.S, Constants.n)
end

export encrypt_to_lwes, decrypt_from_lwes, generate_key

end