module Operations

using Base.Threads
using LinearAlgebra: BLAS

import ..Types: LWE, LWE_ARRAY, pack_lwe, extract_lwe

# Match BLAS threads to Julia threads for maximum throughput
BLAS.set_num_threads(Threads.nthreads())

"""
Homomorphic addition of two LWE ciphertexts.
"""
function add_lwe(lhs::LWE, rhs::LWE)::LWE
    @assert size(lhs) == size(rhs) "LWE ciphertexts must have the same shape"
    a_1, b_1 = extract_lwe(lhs)
    a_2, b_2 = extract_lwe(rhs)
    return pack_lwe(a_1 .+ a_2, b_1 + b_2)
end

"""
Add a plaintext scalar to an LWE ciphertext.
"""
function add_lwe(lhs::LWE, rhs::Float32)::LWE
    a, b = extract_lwe(lhs)
    return pack_lwe(a, b + rhs)
end

"""
Commutative form.
"""
function add_lwe(lhs::Float32, rhs::LWE)::LWE
    return add_lwe(rhs, lhs)
end

"""
Add two arrays of LWE ciphertexts.
"""
function add_lwe(lhs::LWE_ARRAY, rhs::LWE_ARRAY)::LWE_ARRAY
    @assert size(lhs) == size(rhs) "Ciphertext arrays must have the same shape"
    out = Matrix{Float32}(undef, size(lhs))
    return _vector_op!(add_lwe, out, lhs, rhs)
end

"""
Add an array of LWE ciphertexts to a plaintext scalar.
"""
function add_lwe(lhs::LWE_ARRAY, rhs::Float32)::LWE_ARRAY
    out = Matrix{Float32}(undef, size(lhs))
    rhs = fill(rhs, size(lhs, 1))
    return _vector_op!(add_lwe, out, lhs, rhs)
end

"""
Commutative form.
"""
add_lwe(lhs::Float32, rhs::LWE_ARRAY) = add_lwe(rhs, lhs)

"""
Apply a function element-wise to two arrays.
"""
function _vector_op!(f, out::AbstractMatrix, a::AbstractArray, b::AbstractArray)
    n = size(out, 1)

    if n > 100
        @threads for i in 1:n
            lhs_row = @view a[i, :]
            rhs_arg = b isa AbstractMatrix ? (@view b[i, :]) : b[i]
            out[i, :] = f(lhs_row, rhs_arg)
        end
    else
        for i in 1:n
            lhs_row = @view a[i, :]
            rhs_arg = b isa AbstractMatrix ? (@view b[i, :]) : b[i]
            out[i, :] = f(lhs_row, rhs_arg)
        end
    end
    return out
end

# # -----------------------------------------------------------------------------
# # Scalar multiplication
# # -----------------------------------------------------------------------------

# """
#     mult(ciphertexts::Vector{LWE}, scalar::Real) -> Vector{LWE}

# Multiply an array of ciphertexts by a scalar.
# """
# function mult(cts::Vector{LWE}, α::Real)
#     α32 = Float32(α)
#     out = Vector{LWE}(undef, length(cts))

#     if length(cts) > 100
#         @threads for i in eachindex(cts)
#             mask, masked = extract_lwe(cts[i])
#             out[i] = pack_lwe(mask .* α32, masked * α32)
#         end
#     else
#         for i in eachindex(cts)
#             mask, masked = extract_lwe(cts[i])
#             out[i] = pack_lwe(mask .* α32, masked * α32)
#         end
#     end

#     return out
# end

# # -----------------------------------------------------------------------------
# # Encrypted–plaintext dot product
# # -----------------------------------------------------------------------------

# """
#     dot_product(enc_vec, plain_vec, zero_cipher) -> LWE

# Compute ⟨enc_vec, plain_vec⟩ where the first argument is encrypted and the
# second is plaintext.
# """
# function dot_product(enc_vec::AbstractVector{LWE},
#                      plain_vec::AbstractVector{<:Real},
#                      zero_cipher::LWE)::LWE
#     @assert length(enc_vec) == length(plain_vec) "Vectors must have the same length"

#     acc_mask, acc_b = extract_lwe(zero_cipher)  # copies, so we can mutate safely

#     @inbounds for (ct, μ) in zip(enc_vec, plain_vec)
#         μ32 = Float32(μ)
#         if μ32 != 0f0
#             mask, b = extract_lwe(ct)
#             _add_masks!(acc_mask, mask, μ32)
#             acc_b += b * μ32
#         end
#     end

#     return pack_lwe(acc_mask, acc_b)
# end

# """
#     batch_dot_product(enc_batches, plain_batches, zero_cipher) -> Vector{LWE}
# """
# function batch_dot_product(enc_batches::Vector{<:AbstractVector{LWE}},
#                            plain_batches::Vector{<:AbstractVector{<:Real}},
#                            zero_cipher::LWE)
#     @assert length(enc_batches) == length(plain_batches) "Batches must have the same length"
#     n   = length(enc_batches)
#     out = Vector{LWE}(undef, n)

#     @threads for i in 1:n
#         out[i] = dot_product(enc_batches[i], plain_batches[i], zero_cipher)
#     end

#     return out
# end

export add_lwe

end
