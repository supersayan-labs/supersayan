module Linear

using PyCall
import ...Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
import ...Operations: add, mult, dot_product
using Base.Threads

export linear_forward

"""
    linear_forward(input::Vector{T}, weights::AbstractMatrix, bias::Union{AbstractVector, Nothing}=nothing) where T <: Union{LWE, PyObject}

Performs a linear transformation on an encrypted input (batch of LWE ciphertexts).
Optimized with multithreading for batch and output neuron parallelization.

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
    
    # Convert weights to Float64 if needed
    weights_f64 = Matrix{Float64}(weights)
    bias_f64 = bias !== nothing ? Vector{Float64}(bias) : nothing
    
    # Pre-allocate sample inputs to avoid creating views in threads
    sample_inputs = Vector{Vector{LWE}}(undef, batch_size)
    for i in 1:batch_size
        sample_start_idx = (i-1) * in_features + 1
        sample_end_idx = i * in_features
        sample_inputs[i] = input_lwes[sample_start_idx:sample_end_idx]
    end
    
    # Pre-compute weight rows to avoid creating views in threads
    weight_rows = Vector{Vector{Float64}}(undef, out_features)
    for j in 1:out_features
        weight_rows[j] = weights_f64[j, :]
    end
    
    # Process samples sequentially but parallelize neurons
    for i in 1:batch_size
        sample_input = sample_inputs[i]
        
        # Process output neurons in parallel
        if out_features > 1
            local_results = Vector{Tuple{Int, LWE}}(undef, out_features)
            
            @threads for j in 1:out_features
                # Get weights for this output neuron
                weight_row = weight_rows[j]
                
                # Compute dot product
                dp = dot_product(sample_input, weight_row, zero_ciphertext)
                
                # Add bias if provided
                if bias_f64 !== nothing
                    dp = add(dp, bias_f64[j])
                end
                
                # Store result locally with index
                local_results[j] = (j, dp)
            end
            
            # Copy results to output array
            for (j, dp) in local_results
                output_idx = (i-1) * out_features + j
                output[output_idx] = dp
            end
        else
            # Single output neuron case - no need for threading
            for j in 1:out_features
                weight_row = weight_rows[j]
                dp = dot_product(sample_input, weight_row, zero_ciphertext)
                
                if bias_f64 !== nothing
                    dp = add(dp, bias_f64[j])
                end
                
                output_idx = (i-1) * out_features + j
                output[output_idx] = dp
            end
        end
    end
    
    return output
end

"""
    optimized_batch_dot_product(input_slices::Vector{Vector{T}}, weights::AbstractMatrix, bias::Union{AbstractVector, Nothing}) where T <: Union{LWE, PyObject}

Internal helper function that computes dot products for multiple output neurons in parallel.
"""
function optimized_batch_dot_product(input_slices::Vector{Vector{T}}, weights::AbstractMatrix, bias::Union{AbstractVector, Nothing}) where T <: Union{LWE, PyObject}
    batch_size = length(input_slices)
    out_features = size(weights, 1)
    result = Vector{Vector{LWE}}(undef, batch_size)
    
    mask_length = length(first(input_slices)[1].mask)
    zero_ciphertext = LWE(zeros(Float64, mask_length), 0.0)
    
    # Pre-allocate result vectors
    for i in 1:batch_size
        result[i] = Vector{LWE}(undef, out_features)
    end
    
    # Calculate all dot products in parallel
    @threads for j in 1:out_features
        weight_row = view(weights, j, :)
        for i in 1:batch_size
            # Compute dot product
            dp = dot_product(input_slices[i], weight_row, zero_ciphertext)
            
            # Add bias if provided
            if bias !== nothing
                dp = add(dp, bias[j])
            end
            
            result[i][j] = dp
        end
    end
    
    return result
end

end