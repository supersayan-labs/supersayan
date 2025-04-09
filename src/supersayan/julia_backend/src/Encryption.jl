module Encryption

using Random, LinearAlgebra
using PyCall
import ..Constants
import ..Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
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
    encrypt_torus_to_lwe_array(mu::Array{Float64}, shape::Vector{Int64}, s::Vector{Int64}, σ::Float64=σ_LWE)

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
    decrypt_lwe_to_torus(c::Union{LWE, PyObject}, s::Vector{Int64}, p::Int64)

Decrypts an LWE ciphertext `c` with key `s`, projecting the resulting phase to the discrete torus T_p.
Returns a Float64.
"""
function decrypt_lwe_to_torus(c::Union{LWE, PyObject}, s::Vector{Int64}, p::Int64)::Float64
    # Convert PyObject to LWE if needed
    c_lwe = isa(c, PyObject) ? convert_pyobject_to_lwe(c) : c
    
    phase = real_to_torus(c_lwe.masked - dot(s, c_lwe.mask))
    return project_to_discrete_torus(phase, p)
end

"""
    decrypt_lwe_to_torus_vec(ciphertexts::Vector{T}, s::Vector{Int64}, p::Int64) where T <: Union{LWE, PyObject}

Vectorized decryption: decrypts an array of LWE ciphertexts or PyObjects.
Returns an array of Float64.
"""
function decrypt_lwe_to_torus_vec(ciphertexts::Vector{T}, s::Vector{Int64}, p::Int64) where T <: Union{LWE, PyObject}
    # Convert PyObjects to LWE if needed
    if !isempty(ciphertexts) && isa(first(ciphertexts), PyObject)
        ciphertexts_lwe = convert_pyobjects_to_lwes(ciphertexts)
        return [decrypt_lwe_to_torus(c, s, p) for c in ciphertexts_lwe]
    else
        return [decrypt_lwe_to_torus(c, s, p) for c in ciphertexts]
    end
end

"""
    generate_key()

Generates a secret LWE key as a vector of length n with entries in S.
"""
function generate_key()::Vector{Int64}
    return rand(Constants.S, Constants.n)
end

end
