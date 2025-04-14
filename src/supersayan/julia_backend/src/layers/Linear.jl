module Linear

using PyCall
import ...Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
import ...Operations: add, mult, dot_product

export linear_forward

"""
    linear_forward(input::Vector{T}, weights::Matrix{Float64}, bias::Union{Vector{Float64}, Nothing}=nothing) where T <: Union{LWE, PyObject}

Performs a linear transformation on an encrypted input (batch of LWE ciphertexts).

Args:
    input: A vector of LWE ciphertexts (or PyObjects) of length batch_size * in_features.
    weights: A matrix of weights with shape (out_features, in_features).
    bias: An optional vector of bias values with length out_features.

Returns:
    A vector of LWE ciphertexts of length batch_size * out_features.
"""
function linear_forward(input::Vector{T}, weights::AbstractMatrix, bias::Union{AbstractVector, Nothing}=nothing) where T <: Union{LWE, PyObject}
    # Get dimensions
    batch_size = length(input) ÷ size(weights, 2)
    in_features = size(weights, 2)
    out_features = size(weights, 1)
    
    # Check input shape
    if length(input) != batch_size * in_features
        throw(ArgumentError("Input vector length must be batch_size * in_features"))
    end
    
    # Check bias shape if present
    if bias !== nothing && length(bias) != out_features
        throw(ArgumentError("Bias vector length must match out_features"))
    end
    
    # Convert PyObjects to LWE if needed
    input_lwes = isa(first(input), PyObject) ? convert_pyobjects_to_lwes(input) : input
    
    # Prepare output vector
    output = Vector{LWE}(undef, batch_size * out_features)
    
    # Get a zero ciphertext for dot product initialization
    mask_length = length(input_lwes[1].mask)
    zero_ciphertext = LWE(zeros(Float64, mask_length), 0.0)
    
    # Process each sample in the batch
    for i in 1:batch_size
        # Get the input slice for this sample
        sample_start_idx = (i-1) * in_features + 1
        sample_end_idx = i * in_features
        sample_input = input_lwes[sample_start_idx:sample_end_idx]
        
        # Process each output neuron
        for j in 1:out_features
            # Get weights for this output neuron
            weight_row = weights[j, :]
            
            # Compute dot product
            dp = dot_product(sample_input, weight_row, zero_ciphertext)
            
            # Add bias if provided
            if bias !== nothing
                dp = add(dp, bias[j])
            end
            
            # Store result
            output_idx = (i-1) * out_features + j
            output[output_idx] = dp
        end
    end
    
    return output
end

end