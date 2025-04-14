module Operations

import ..Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
using PyCall
using Base.Threads
using LinearAlgebra: BLAS

export add, mult, dot_product, batch_dot_product, batch_add

# Set BLAS threads to match Julia threads for optimal performance
BLAS.set_num_threads(Threads.nthreads())

"""
    add(lhs::Vector{T}, rhs::Vector{T}) where T <: Union{LWE, PyObject}

Elementwise addition of two arrays of LWE ciphertexts.
Optimized with multithreading for large arrays.
"""
function add(lhs::Vector{T}, rhs::Vector{T}) where T <: Union{LWE, PyObject}
    if length(lhs) != length(rhs)
        throw(ArgumentError("Ciphertext arrays must have the same length"))
    end
    
    # Convert PyObjects to LWE if needed
    lhs_lwes = isa(first(lhs), PyObject) ? convert_pyobjects_to_lwes(lhs) : lhs
    rhs_lwes = isa(first(rhs), PyObject) ? convert_pyobjects_to_lwes(rhs) : rhs
    
    result = Vector{LWE}(undef, length(lhs))
    
    # Use multithreading for large arrays
    if length(lhs) > 100  # Threshold for parallelization
        @threads for i in 1:length(lhs)
            a_new = lhs_lwes[i].mask .+ rhs_lwes[i].mask
            b_new = lhs_lwes[i].masked + rhs_lwes[i].masked
            result[i] = LWE(a_new, b_new)
        end
    else
        # Sequential for small arrays to avoid thread overhead
        for i in 1:length(lhs)
            a_new = lhs_lwes[i].mask .+ rhs_lwes[i].mask
            b_new = lhs_lwes[i].masked + rhs_lwes[i].masked
            result[i] = LWE(a_new, b_new)
        end
    end
    
    return result
end

"""
    batch_add(lhs_batch::Vector{Vector{T}}, rhs_batch::Vector{Vector{T}}) where T <: Union{LWE, PyObject}

Performs batch addition on multiple arrays in parallel.
"""
function batch_add(lhs_batch::Vector{Vector{T}}, rhs_batch::Vector{Vector{T}}) where T <: Union{LWE, PyObject}
    if length(lhs_batch) != length(rhs_batch)
        throw(ArgumentError("Batch arrays must have the same length"))
    end
    
    result = Vector{Vector{LWE}}(undef, length(lhs_batch))
    
    @threads for batch_idx in 1:length(lhs_batch)
        result[batch_idx] = add(lhs_batch[batch_idx], rhs_batch[batch_idx])
    end
    
    return result
end

"""
    add(lhs::Vector{T}, rhs::Real) where T <: Union{LWE, PyObject}

Adds a scalar rhs to each LWE ciphertext in the array lhs.
Optimized with multithreading for large arrays.
"""
function add(lhs::Vector{T}, rhs::Real) where T <: Union{LWE, PyObject}
    # Convert PyObjects to LWE if needed
    lhs_lwes = isa(first(lhs), PyObject) ? convert_pyobjects_to_lwes(lhs) : lhs
    
    result = Vector{LWE}(undef, length(lhs))
    
    # Use multithreading for large arrays
    if length(lhs) > 100  # Threshold for parallelization
        @threads for i in 1:length(lhs)
            result[i] = LWE(lhs_lwes[i].mask, lhs_lwes[i].masked + rhs)
        end
    else
        # Sequential for small arrays
        for i in 1:length(lhs)
            result[i] = LWE(lhs_lwes[i].mask, lhs_lwes[i].masked + rhs)
        end
    end
    
    return result
end

"""
    add(lhs::T, rhs::Real) where T <: Union{LWE, PyObject}

Adds a scalar rhs to a single LWE ciphertext lhs.
Returns a single LWE ciphertext.
"""
function add(lhs::T, rhs::Real) where T <: Union{LWE, PyObject}
    # Convert PyObject to LWE if needed
    lhs_lwe = isa(lhs, PyObject) ? convert_pyobject_to_lwe(lhs) : lhs
    
    # Add the scalar to the b term
    return LWE(lhs_lwe.mask, lhs_lwe.masked + rhs)
end

"""
    mult(lhs::Vector{T}, scalar::Real) where T <: Union{LWE, PyObject}

Multiplies each LWE ciphertext in lhs by the scalar.
Returns an array of LWE ciphertexts.
Optimized with multithreading for large arrays.
"""
function mult(lhs::Vector{T}, scalar::Real) where T <: Union{LWE, PyObject}
    # Convert PyObjects to LWE if needed
    lhs_lwes = isa(first(lhs), PyObject) ? convert_pyobjects_to_lwes(lhs) : lhs
    
    result = Vector{LWE}(undef, length(lhs))
    
    # Use multithreading for large arrays
    if length(lhs) > 100  # Threshold for parallelization
        @threads for i in 1:length(lhs)
            # Use broadcast (dot operator) for vectorized operation
            result[i] = LWE(lhs_lwes[i].mask .* scalar, lhs_lwes[i].masked * scalar)
        end
    else
        # Sequential for small arrays
        for i in 1:length(lhs)
            result[i] = LWE(lhs_lwes[i].mask .* scalar, lhs_lwes[i].masked * scalar)
        end
    end
    
    return result
end

"""
    dot_product(encrypted_vector::Vector{T}, plain_vector::AbstractVector{<:Real}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}

Computes the encrypted dot product between an encrypted vector and a plaintext vector.
Optimized implementation that reduces allocations and uses SIMD where possible.

Returns an LWE ciphertext representing the dot product.
"""
function dot_product(encrypted_vector::Vector{T}, plain_vector::AbstractVector{<:Real}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    if length(encrypted_vector) != length(plain_vector)
        throw(ArgumentError("Vectors must have the same length"))
    end
    
    # Convert PyObjects to LWE
    enc_lwes = isa(first(encrypted_vector), PyObject) ? convert_pyobjects_to_lwes(encrypted_vector) : encrypted_vector
    z = isa(zero_ciphertext, PyObject) ? convert_pyobject_to_lwe(zero_ciphertext) : zero_ciphertext
    
    # Start with a copy of zero ciphertext
    mask_length = length(z.mask)
    result_mask = copy(z.mask)
    result_masked = z.masked
    
    # Convert plain vector to Float64 for best performance
    plain_vector_f64 = Vector{Float64}(plain_vector)
    
    # Process chunks of the input for cache efficiency
    chunk_size = 16  # Tuned for typical CPU cache lines
    num_chunks = div(length(enc_lwes), chunk_size)
    remainder = rem(length(enc_lwes), chunk_size)
    
    # Process each chunk
    for chunk in 1:num_chunks
        start_idx = (chunk - 1) * chunk_size + 1
        end_idx = chunk * chunk_size
        
        # Accumulate within chunk
        for i in start_idx:end_idx
            scalar = plain_vector_f64[i]
            # Skip zero multiplications
            if scalar != 0
                # Accumulate directly into result
                @simd for j in 1:mask_length
                    result_mask[j] += enc_lwes[i].mask[j] * scalar
                end
                result_masked += enc_lwes[i].masked * scalar
            end
        end
    end
    
    # Process remainder
    if remainder > 0
        start_idx = num_chunks * chunk_size + 1
        for i in start_idx:length(enc_lwes)
            scalar = plain_vector_f64[i]
            if scalar != 0
                @simd for j in 1:mask_length
                    result_mask[j] += enc_lwes[i].mask[j] * scalar
                end
                result_masked += enc_lwes[i].masked * scalar
            end
        end
    end
    
    return LWE(result_mask, result_masked)
end

# Add specialized method for SubArray to handle views correctly for encrypted_vector
function dot_product(encrypted_vector::SubArray{T, 1}, plain_vector::AbstractVector{<:Real}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    # Convert SubArray to a regular Vector
    enc_vec = collect(encrypted_vector)
    # Then call the regular method
    return dot_product(enc_vec, plain_vector, zero_ciphertext)
end

# Add specialized method for SubArray to handle views correctly for plain_vector
function dot_product(encrypted_vector::Vector{T}, plain_vector::SubArray{<:Real, 1}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    # Convert SubArray to a regular Vector
    plain_vec = collect(plain_vector)
    # Then call the regular method
    return dot_product(encrypted_vector, plain_vec, zero_ciphertext)
end

# Add specialized method for both SubArrays
function dot_product(encrypted_vector::SubArray{T, 1}, plain_vector::SubArray{<:Real, 1}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    # Convert both SubArrays to regular Vectors
    enc_vec = collect(encrypted_vector)
    plain_vec = collect(plain_vector)
    # Then call the regular method
    return dot_product(enc_vec, plain_vec, zero_ciphertext)
end

"""
    batch_dot_product(encrypted_vectors::Vector{Vector{T}}, plain_vectors::Vector{Vector{<:Real}}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}

Computes multiple dot products in parallel.
"""
function batch_dot_product(encrypted_vectors::Vector{Vector{T}}, plain_vectors::Vector{Vector{<:Real}}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    if length(encrypted_vectors) != length(plain_vectors)
        throw(ArgumentError("Batch arrays must have the same length"))
    end
    
    result = Vector{LWE}(undef, length(encrypted_vectors))
    
    @threads for batch_idx in 1:length(encrypted_vectors)
        result[batch_idx] = dot_product(encrypted_vectors[batch_idx], plain_vectors[batch_idx], zero_ciphertext)
    end
    
    return result
end

end
