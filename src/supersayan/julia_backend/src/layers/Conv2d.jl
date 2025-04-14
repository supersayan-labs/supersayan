module Conv2d

using PyCall
import ...Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
import ...Operations: add, mult, dot_product

export conv2d_forward, build_toeplitz_matrices

"""
    compute_output_shape(input_shape::Tuple{Int, Int}, kernel_size::Tuple{Int, Int}, 
                         stride::Tuple{Int, Int}, padding::Tuple{Int, Int}, dilation::Tuple{Int, Int})

Computes the output shape of a convolution operation.

Args:
    input_shape: A tuple (height, width) of the input feature map.
    kernel_size: A tuple (height, width) of the kernel.
    stride: A tuple (height, width) for the stride.
    padding: A tuple (height, width) for the padding.
    dilation: A tuple (height, width) for the dilation.

Returns:
    A tuple (height, width) of the output feature map.
"""
function compute_output_shape(input_shape::Tuple{Int, Int}, kernel_size::Tuple{Int, Int}, 
                             stride::Tuple{Int, Int}, padding::Tuple{Int, Int}, dilation::Tuple{Int, Int})
    h_in, w_in = input_shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation
    
    h_out = Int(floor((h_in + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1))
    w_out = Int(floor((w_in + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1))
    
    return (h_out, w_out)
end

"""
    build_toeplitz_matrices(input_shape::Tuple{Int, Int, Int, Int}, weight::Array{Float64, 4}, 
                           kernel_size::Tuple{Int, Int}, stride::Tuple{Int, Int}, 
                           padding::Tuple{Int, Int}, dilation::Tuple{Int, Int}, groups::Int)

Builds the Toeplitz matrices for a convolution operation.

Args:
    input_shape: A tuple (batch, channels, height, width) of the input.
    weight: A 4D array of weights with shape (out_channels, in_channels // groups, kernel_height, kernel_width).
    kernel_size: A tuple (height, width) of the kernel.
    stride: A tuple (height, width) for the stride.
    padding: A tuple (height, width) for the padding.
    dilation: A tuple (height, width) for the dilation.
    groups: An integer specifying the groups for grouped convolution.

Returns:
    A vector of matrices, one for each output channel.
"""
function build_toeplitz_matrices(input_shape::Tuple{Int, Int, Int, Int}, weight::AbstractArray, 
                               kernel_size::Tuple{Int, Int}, stride::Tuple{Int, Int}, 
                               padding::Tuple{Int, Int}, dilation::Tuple{Int, Int}, groups::Int)
    _, in_channels, h_in, w_in = input_shape
    out_channels, in_channels_per_group, k_h, k_w = size(weight)
    
    # Compute output dimensions
    h_out, w_out = compute_output_shape((h_in, w_in), kernel_size, stride, padding, dilation)
    
    # Prepare to store Toeplitz matrices
    toeplitz_matrices = Vector{Matrix{Float64}}(undef, out_channels)
    
    # For each output channel, create a Toeplitz matrix
    for out_c in 1:out_channels
        # For grouped convolutions, determine which input channels to use
        in_c_start = ((out_c-1) ÷ (out_channels ÷ groups)) * (in_channels ÷ groups) + 1
        in_c_end = in_c_start + (in_channels ÷ groups) - 1
        
        # Create the Toeplitz matrix for this output channel
        # Shape: (h_out * w_out, h_in * w_in * (in_channels // groups))
        toeplitz = zeros(h_out * w_out, h_in * w_in * (in_channels ÷ groups))
        
        # Fill the Toeplitz matrix based on convolution parameters
        for i in 1:h_out
            for j in 1:w_out
                out_idx = (i-1) * w_out + j
                
                for di in 1:k_h
                    for dj in 1:k_w
                        # Compute input positions with stride, padding, and dilation
                        in_i = (i-1) * stride[1] + (di-1) * dilation[1] - padding[1] + 1
                        in_j = (j-1) * stride[2] + (dj-1) * dilation[2] - padding[2] + 1
                        
                        # Skip if the indices are outside the input boundaries
                        if 1 <= in_i <= h_in && 1 <= in_j <= w_in
                            # For each input channel in this group
                            for in_c_idx in 1:(in_channels ÷ groups)
                                in_c = in_c_start + in_c_idx - 1
                                in_idx = (in_c_idx-1) * (h_in * w_in) + (in_i-1) * w_in + in_j
                                
                                # Get the weight value
                                weight_idx = (in_c_idx, di, dj)
                                toeplitz[out_idx, in_idx] = weight[out_c, weight_idx...]
                            end
                        end
                    end
                end
            end
        end
        
        toeplitz_matrices[out_c] = toeplitz
    end
    
    return toeplitz_matrices
end

"""
    conv2d_forward(input::Vector{T}, input_shape::Tuple{Int, Int, Int, Int}, 
                  toeplitz_matrices::Vector{Matrix{Float64}}, bias::Union{Vector{Float64}, Nothing}=nothing) where T <: Union{LWE, PyObject}

Performs the forward pass of a 2D convolution using homomorphic operations.

Args:
    input: A vector of LWE ciphertexts (or PyObjects) representing the flattened input.
    input_shape: A tuple (batch, channels, height, width) indicating the shape of the input.
    toeplitz_matrices: A vector of matrices, one for each output channel.
    bias: An optional vector of bias values, one for each output channel.

Returns:
    A vector of LWE ciphertexts representing the flattened output.
"""
function conv2d_forward(input::Vector{T}, input_shape::Tuple{Int, Int, Int, Int}, 
                      toeplitz_matrices::Vector, bias::Union{AbstractVector, Nothing}=nothing) where T <: Union{LWE, PyObject}
    batch_size, in_channels, h_in, w_in = input_shape
    out_channels = length(toeplitz_matrices)
    
    # Get the shape of the first Toeplitz matrix to determine output size
    m, _ = size(toeplitz_matrices[1])
    h_out = Int(sqrt(m))  # Assuming square output for simplicity
    w_out = h_out
    
    # Convert PyObjects to LWE if needed
    input_lwes = isa(first(input), PyObject) ? convert_pyobjects_to_lwes(input) : input
    
    # Get a zero ciphertext for dot product initialization
    mask_length = length(input_lwes[1].mask)
    zero_ciphertext = LWE(zeros(Float64, mask_length), 0.0)
    
    # Prepare output vector
    output = Vector{LWE}(undef, batch_size * out_channels * h_out * w_out)
    
    # For each sample in the batch
    for b in 1:batch_size
        # Get the slice of the input for this batch
        batch_start_idx = (b-1) * in_channels * h_in * w_in + 1
        batch_end_idx = b * in_channels * h_in * w_in
        input_vector = input_lwes[batch_start_idx:batch_end_idx]
        
        # For each output channel
        for out_c in 1:out_channels
            # Get the Toeplitz matrix for this channel
            toeplitz = toeplitz_matrices[out_c]
            
            # For each output position
            for i in 1:h_out
                for j in 1:w_out
                    out_idx = (i-1) * w_out + j
                    toeplitz_row = toeplitz[out_idx, :]
                    
                    # Compute dot product
                    result = dot_product(input_vector, toeplitz_row, zero_ciphertext)
                    
                    # Add bias if present
                    if bias !== nothing
                        result = add(result, bias[out_c])
                    end
                    
                    # Store in output array
                    output_idx = (b-1) * out_channels * h_out * w_out + (out_c-1) * h_out * w_out + (i-1) * w_out + j
                    output[output_idx] = result
                end
            end
        end
    end
    
    return output
end

end