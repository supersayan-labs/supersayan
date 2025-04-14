module Operations

import ..Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
using PyCall
export add, mult, dot_product

"""
    add(lhs::Vector{T}, rhs::Vector{T}) where T <: Union{LWE, PyObject}

Elementwise addition of two arrays of LWE ciphertexts.
"""
function add(lhs::Vector{T}, rhs::Vector{T}) where T <: Union{LWE, PyObject}
    if length(lhs) != length(rhs)
        throw(ArgumentError("Ciphertext arrays must have the same length"))
    end
    
    # Convert PyObjects to LWE if needed
    lhs_lwes = isa(first(lhs), PyObject) ? convert_pyobjects_to_lwes(lhs) : lhs
    rhs_lwes = isa(first(rhs), PyObject) ? convert_pyobjects_to_lwes(rhs) : rhs
    
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        a_new = lhs_lwes[i].mask .+ rhs_lwes[i].mask
        b_new = lhs_lwes[i].masked + rhs_lwes[i].masked
        result[i] = LWE(a_new, b_new)
    end
    return result
end

"""
    add(lhs::Vector{T}, rhs::Float64) where T <: Union{LWE, PyObject}

Adds a scalar rhs to each LWE ciphertext in the array lhs.
"""
function add(lhs::Vector{T}, rhs::Real) where T <: Union{LWE, PyObject}
    # Convert PyObjects to LWE if needed
    lhs_lwes = isa(first(lhs), PyObject) ? convert_pyobjects_to_lwes(lhs) : lhs
    
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        result[i] = LWE(lhs_lwes[i].mask, lhs_lwes[i].masked + rhs)
    end
    return result
end

"""
    add(lhs::T, rhs::Float64) where T <: Union{LWE, PyObject}

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
    mult(lhs::Vector{T}, scalar::Float64) where T <: Union{LWE, PyObject}

Multiplies each LWE ciphertext in lhs by the scalar.
Returns an array of LWE ciphertexts.
"""
function mult(lhs::Vector{T}, scalar::Real) where T <: Union{LWE, PyObject}
    # Convert PyObjects to LWE if needed
    lhs_lwes = isa(first(lhs), PyObject) ? convert_pyobjects_to_lwes(lhs) : lhs
    
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        result[i] = LWE(lhs_lwes[i].mask .* scalar, lhs_lwes[i].masked * scalar)
    end
    return result
end

"""
    dot_product(encrypted_vector::Vector{T}, plain_vector::Vector{Float64}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}

Computes the encrypted dot product between an encrypted vector and a plaintext vector.
The function uses the defined multiplication (mult) and addition (add) operations.
The accumulation starts from the provided zero_ciphertext.

Returns an LWE ciphertext representing the dot product.
"""
function dot_product(encrypted_vector::Vector{T}, plain_vector::AbstractVector{<:Real}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    if length(encrypted_vector) != length(plain_vector)
        throw(ArgumentError("Vectors must have the same length"))
    end
    
    # Convert PyObjects to LWE
    enc_lwes = isa(first(encrypted_vector), PyObject) ? convert_pyobjects_to_lwes(encrypted_vector) : encrypted_vector
    z = isa(zero_ciphertext, PyObject) ? convert_pyobject_to_lwe(zero_ciphertext) : zero_ciphertext
    
    # Initialize result with zero ciphertext
    result = [z]
    
    for i in 1:length(enc_lwes)
        # Multiply the i-th encrypted element by the corresponding scalar
        term = mult([enc_lwes[i]], plain_vector[i])
        
        # Add the term to the current result
        result = add(result, term)
    end
    
    return result[1]
end

end
