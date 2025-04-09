module Operations

import ..Types: LWE
using PyCall  # <-- NEW: Import PyCall to unwrap PyObjects
export add, mult, dot_product

"""
    add(lhs::Vector{LWE}, rhs::Vector{LWE})

Elementwise addition of two arrays of LWE ciphertexts.
"""
function add(lhs::Vector{LWE}, rhs::Vector{LWE})
    if length(lhs) != length(rhs)
        throw(ArgumentError("Ciphertext arrays must have the same length"))
    end
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        a_new = lhs[i].mask .+ rhs[i].mask
        b_new = lhs[i].masked + rhs[i].masked
        result[i] = LWE(a_new, b_new)
    end
    return result
end

"""
    add(lhs::Vector{LWE}, rhs::LWE)

Adds a single LWE ciphertext to each ciphertext in the array.
"""
function add(lhs::Vector{LWE}, rhs::LWE)
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        a_new = lhs[i].mask .+ rhs.mask
        b_new = lhs[i].masked + rhs.masked
        result[i] = LWE(a_new, b_new)
    end
    return result
end

"""
    add(lhs::Vector{LWE}, rhs::Float64)

Adds a scalar rhs to each LWE ciphertext in the array lhs.
"""
function add(lhs::Vector{LWE}, rhs::Float64)
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        result[i] = LWE(lhs[i].mask, lhs[i].masked + rhs)
    end
    return result
end

"""
    add(lhs::Vector{PyObject}, rhs::Float64)

Adds a scalar rhs to each PyObject-wrapped LWE ciphertext in the array lhs.
"""
function add(lhs::Vector{PyObject}, rhs::Float64)
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        # Extract mask and masked value from PyObject
        mask = lhs[i].mask
        masked = lhs[i].masked
        # Create a new LWE with the added scalar
        result[i] = LWE(mask, masked + rhs)
    end
    return result
end

"""
    mult(lhs::Vector{LWE}, scalar::Float64)

Multiplies each LWE ciphertext in lhs by the scalar.
Returns an array of LWE ciphertexts.
"""
function mult(lhs::Vector{LWE}, scalar::Float64)
    result = Vector{LWE}(undef, length(lhs))
    for i in 1:length(lhs)
        result[i] = LWE(lhs[i].mask .* scalar, lhs[i].masked * scalar)
    end
    return result
end

"""
    dot_product(encrypted_vector::Vector{LWE}, plain_vector::Vector{Float64}, zero_ciphertext::LWE)

Computes the encrypted dot product between an encrypted vector (list of LWE ciphertexts)
and a plaintext vector. The function assumes that multiplication (mult) and addition (add) 
over ciphertexts are defined. The accumulation starts from the provided zero_ciphertext.

Returns an LWE ciphertext representing the dot product.
"""
function dot_product(encrypted_vector::Vector{LWE}, plain_vector::Vector{Float64}, zero_ciphertext::LWE)
    if length(encrypted_vector) != length(plain_vector)
        throw(ArgumentError("Vectors must have the same length"))
    end
    result = zero_ciphertext
    for i in 1:length(encrypted_vector)
        # Multiply the i-th encrypted element by the corresponding scalar.
        term = mult([encrypted_vector[i]], plain_vector[i])[1]
        # Add the term to the current result.
        result = LWE(result.mask .+ term.mask, result.masked + term.masked)
    end
    return result
end

# New method that properly handles PyObjects
function dot_product(encrypted_vector::Vector{PyObject}, plain_vector::Vector{Float64}, zero_ciphertext::PyObject)
    # Create a vector of Any type (not pre-typed as LWE)
    lwes = Vector{Any}(undef, length(encrypted_vector))
    
    # Extract data from PyObjects and create new LWE objects
    for i in 1:length(encrypted_vector)
        py_obj = encrypted_vector[i]
        # Get mask and masked value from PyObject
        mask = py_obj.mask
        masked = py_obj.masked
        # Create a new LWE object with this data
        lwes[i] = LWE(mask, masked)
    end
    
    # Do the same for zero_ciphertext
    z_mask = zero_ciphertext.mask
    z_masked = zero_ciphertext.masked
    z = LWE(z_mask, z_masked)
    
    # Call the original dot_product with native LWE objects
    return dot_product(convert(Vector{LWE}, lwes), plain_vector, z)
end

"""
    dot_product(encrypted_vector::Vector{LWE}, plain_vector::Vector{Float64}, zero_ciphertext::PyObject)

Handles case where encrypted vector contains LWE objects but zero_ciphertext is a PyObject.
"""
function dot_product(encrypted_vector::Vector{LWE}, plain_vector::Vector{Float64}, zero_ciphertext::PyObject)
    # Extract mask and masked value from PyObject zero_ciphertext
    z_mask = zero_ciphertext.mask
    z_masked = zero_ciphertext.masked
    z = LWE(z_mask, z_masked)
    
    # Call the original dot_product with native LWE objects
    return dot_product(encrypted_vector, plain_vector, z)
end

end
