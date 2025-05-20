module Operations

# -----------------------------------------------------------------------------
# Imports and exports
# -----------------------------------------------------------------------------

import ..Types: LWE, pack_lwe, extract_lwe

using Base.Threads
using LinearAlgebra: BLAS

export add, mult, dot_product, batch_dot_product, batch_add

# Match BLAS threads to Julia threads for maximum throughput
BLAS.set_num_threads(Threads.nthreads())

# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

"""
    _add_masks!(dst, src, α = 1f0)

In‑place add `α*src` to `dst`.  `dst` is modified and returned.
"""
@inline function _add_masks!(dst::AbstractVector{Float32}, src::AbstractVector{Float32}, α::Float32 = 1f0)
    @inbounds @simd for i in eachindex(dst)
        dst[i] += src[i] * α
    end
    return dst
end

# -----------------------------------------------------------------------------
# Element‑wise addition
# -----------------------------------------------------------------------------

"""
    add(lhs::LWE, rhs::LWE) -> LWE

Homomorphic addition of two ciphertexts.
"""
function add(lhs::LWE, rhs::LWE)::LWE
    m₁, b₁ = extract_lwe(lhs)
    m₂, b₂ = extract_lwe(rhs)
    return pack_lwe(m₁ .+ m₂, b₁ + b₂)
end

"""
    add(ciphertext::LWE, plaintext::Real) -> LWE

Add a plaintext scalar to a ciphertext.
"""
function add(ct::LWE, μ::Real)::LWE
    mask, masked = extract_lwe(ct)
    return pack_lwe(mask, masked + Float32(μ))
end

"""
    add(plaintext::Real, ciphertext::LWE) -> LWE

Commutative form.
"""
add(μ::Real, ct::LWE) = add(ct, μ)

# -----------------------------------------------------------------------------
# Vectorised addition helpers
# -----------------------------------------------------------------------------

function _vector_op!(f, out::Vector{LWE}, a::Vector, b)
    n = length(out)
    if n > 100
        @threads for i in 1:n
            out[i] = f(a[i], b[i])
        end
    else
        for i in 1:n
            out[i] = f(a[i], b[i])
        end
    end
    return out
end

"""
    add(lhs::Vector{LWE}, rhs::Vector{LWE}) -> Vector{LWE}
"""
function add(lhs::Vector{LWE}, rhs::Vector{LWE})
    @assert length(lhs) == length(rhs) "Ciphertext arrays must have the same length"
    out = Vector{LWE}(undef, length(lhs))
    return _vector_op!(add, out, lhs, rhs)
end

"""
    add(ciphertexts::Vector{LWE}, plaintext::Real) -> Vector{LWE}
"""
function add(cts::Vector{LWE}, μ::Real)
    out = Vector{LWE}(undef, length(cts))
    if length(cts) > 100
        @threads for i in eachindex(cts)
            out[i] = add(cts[i], μ)
        end
    else
        for i in eachindex(cts)
            out[i] = add(cts[i], μ)
        end
    end
    return out
end

add(μ::Real, cts::Vector{LWE}) = add(cts, μ)

"""
    batch_add(lhs_batch, rhs_batch) -> Vector{Vector{LWE}}
"""
function batch_add(lhs_batch::Vector{<:AbstractVector{LWE}},
                   rhs_batch::Vector{<:AbstractVector{LWE}})
    @assert length(lhs_batch) == length(rhs_batch) "Batches must have the same length"
    n   = length(lhs_batch)
    out = Vector{Vector{LWE}}(undef, n)

    @threads for i in 1:n
        out[i] = add(lhs_batch[i], rhs_batch[i])
    end

    return out
end

# -----------------------------------------------------------------------------
# Scalar multiplication
# -----------------------------------------------------------------------------

"""
    mult(ciphertexts::Vector{LWE}, scalar::Real) -> Vector{LWE}

Multiply an array of ciphertexts by a scalar.
"""
function mult(cts::Vector{LWE}, α::Real)
    α32 = Float32(α)
    out = Vector{LWE}(undef, length(cts))

    if length(cts) > 100
        @threads for i in eachindex(cts)
            mask, masked = extract_lwe(cts[i])
            out[i] = pack_lwe(mask .* α32, masked * α32)
        end
    else
        for i in eachindex(cts)
            mask, masked = extract_lwe(cts[i])
            out[i] = pack_lwe(mask .* α32, masked * α32)
        end
    end

    return out
end

# -----------------------------------------------------------------------------
# Encrypted–plaintext dot product
# -----------------------------------------------------------------------------

"""
    dot_product(enc_vec, plain_vec, zero_cipher) -> LWE

Compute ⟨enc_vec, plain_vec⟩ where the first argument is encrypted and the
second is plaintext.
"""
function dot_product(enc_vec::AbstractVector{LWE},
                     plain_vec::AbstractVector{<:Real},
                     zero_cipher::LWE)::LWE
    @assert length(enc_vec) == length(plain_vec) "Vectors must have the same length"

    acc_mask, acc_b = extract_lwe(zero_cipher)  # copies, so we can mutate safely

    @inbounds for (ct, μ) in zip(enc_vec, plain_vec)
        μ32 = Float32(μ)
        if μ32 != 0f0
            mask, b = extract_lwe(ct)
            _add_masks!(acc_mask, mask, μ32)
            acc_b += b * μ32
        end
    end

    return pack_lwe(acc_mask, acc_b)
end

"""
    batch_dot_product(enc_batches, plain_batches, zero_cipher) -> Vector{LWE}
"""
function batch_dot_product(enc_batches::Vector{<:AbstractVector{LWE}},
                           plain_batches::Vector{<:AbstractVector{<:Real}},
                           zero_cipher::LWE)
    @assert length(enc_batches) == length(plain_batches) "Batches must have the same length"
    n   = length(enc_batches)
    out = Vector{LWE}(undef, n)

    @threads for i in 1:n
        out[i] = dot_product(enc_batches[i], plain_batches[i], zero_cipher)
    end

    return out
end

end # module
