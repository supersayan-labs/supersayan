module Encryption

using Random, LinearAlgebra
using PyCall  # Add PyCall import
import ..Constants
import ..Types: LWE
export encrypt_torus_to_lwe, encrypt_torus_to_lwe_vec, decrypt_lwe_to_torus, decrypt_lwe_to_torus_vec, generate_key

# Helper function: maps a real number to its representative in the torus [-1/2, 1/2)
function real_to_torus(μ::Float64)
    μ = mod(μ, 1.0)
    return μ - (μ >= 0.5 ? 1.0 : 0.0)
end

# Helper: projects μ to the discrete torus defined by p.
function project_to_discrete_torus(μ::Float64, p::Int64)
    return real_to_torus(round(μ * p) / p)
end

"""
    encrypt_torus_to_lwe(μ::Float64, s::Vector{Int64}, σ::Float64=σ_LWE)

Encrypts a scalar torus element μ with secret key `s` and noise standard deviation `σ`.
Returns an LWE ciphertext.
"""
function encrypt_torus_to_lwe(μ::Float64, s::Vector{Int64}, σ::Float64=Constants.σ_LWE)::LWE
    # Generate a random mask vector; here we "encrypt" by mapping uniform randoms into the torus.
    a = real_to_torus.(rand(length(s)))
    e = σ * randn(Float64)
    b = real_to_torus(μ + dot(s, a) + e)
    return LWE(a, b)
end

"""
    encrypt_torus_to_lwe_vec(mu::Vector{Float64}, s::Vector{Int64}, σ::Float64=σ_LWE)

Vectorized encryption: encrypts each element in the array `mu` using the same key `s`.
Returns an array of LWE ciphertexts.
"""
function encrypt_torus_to_lwe_vec(mu::Vector{Float64}, s::Vector{Int64}, σ::Float64=Constants.σ_LWE)
    return [encrypt_torus_to_lwe(m, s, σ) for m in mu]
end

"""
    encrypt_torus_to_lwe_array(mu::Array{Float64}, s::Vector{Int64}, σ::Float64=σ_LWE)

Multi-dimensional array encryption: encrypts each element in the array `mu` using the same key `s`,
preserving the original array shape.
Returns an array of LWE ciphertexts with the same shape as `mu`.
"""
function encrypt_torus_to_lwe_array(mu::Array{Float64}, shape::Vector{Int64}, s::Vector{Int64}, σ::Float64=Constants.σ_LWE)
    # Flatten the array, encrypt each element, then reshape back to original shape
    flattened = vec(mu)
    encrypted_flat = encrypt_torus_to_lwe_vec(flattened, s, σ)
    
    # Return both the encrypted array and the original shape
    return encrypted_flat, shape
end

"""
    decrypt_lwe_to_torus(c::LWE, s::Vector{Int64}, p::Int64)

Decrypts an LWE ciphertext `c` with key `s`, projecting the resulting phase to the discrete torus T_p.
Returns a Float64.
"""
function decrypt_lwe_to_torus(c::LWE, s::Vector{Int64}, p::Int64)::Float64
    phase = real_to_torus(c.masked - dot(s, c.mask))
    return project_to_discrete_torus(phase, p)
end

"""
    decrypt_lwe_to_torus_vec(ciphertexts::Vector{LWE}, s::Vector{Int64}, p::Int64)

Vectorized decryption: decrypts an array of LWE ciphertexts.
Returns an array of Float64.
"""
function decrypt_lwe_to_torus_vec(ciphertexts::Vector{LWE}, s::Vector{Int64}, p::Int64)
    return [decrypt_lwe_to_torus(c, s, p) for c in ciphertexts]
end

"""
    decrypt_lwe_to_torus_vec(ciphertexts::Vector{PyObject}, s::Vector{Int64}, p::Int64)

Vectorized decryption for PyObjects: converts PyObjects to LWE objects before decryption.
Returns an array of Float64.
"""
function decrypt_lwe_to_torus_vec(ciphertexts::Vector{PyObject}, s::Vector{Int64}, p::Int64)
    # Convert each PyObject to LWE by extracting the mask and masked values
    lwe_ciphertexts = Vector{LWE}(undef, length(ciphertexts))
    for i in 1:length(ciphertexts)
        mask = ciphertexts[i].mask
        masked = ciphertexts[i].masked
        lwe_ciphertexts[i] = LWE(mask, masked)
    end
    
    # Call the original function with LWE objects
    return decrypt_lwe_to_torus_vec(lwe_ciphertexts, s, p)
end

"""
    generate_key()

Generates a secret LWE key as a vector of length n with entries in S.
"""
function generate_key()::Vector{Int64}
    return rand(Constants.S, Constants.n)
end

end
