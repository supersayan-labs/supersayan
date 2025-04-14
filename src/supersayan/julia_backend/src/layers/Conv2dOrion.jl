module Conv2dOrion

using PyCall
import ...Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
import ...Operations: add, mult, dot_product
using Base.Threads
using LinearAlgebra: BLAS

# Set BLAS threads to match Julia threads for optimal performance
BLAS.set_num_threads(Threads.nthreads())

export build_encoding_plan, conv2d_orion_forward

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
    compute_bsgs_factors(n::Int)

Computes optimal baby-step giant-step factors for a given dimension.

Args:
    n: The dimension to factor for BSGS optimization.

Returns:
    A tuple (n1, n2) where n1 * n2 >= n and n1 and n2 are as close as possible to sqrt(n).
"""
function compute_bsgs_factors(n::Int)
    n1 = Int(ceil(sqrt(n)))
    n2 = Int(ceil(n / n1))
    
    # Ensure n1 * n2 >= n
    while n1 * n2 < n
        n2 += 1
    end
    
    return (n1, n2)
end

"""
    get_stride_permutation(h_out::Int, w_out::Int, stride::Tuple{Int, Int})

Generates index permutation for single-shot multiplexing with stride > 1.

Args:
    h_out: Output height
    w_out: Output width
    stride: A tuple (stride_h, stride_w) for the stride.

Returns:
    A vector of indices representing the permutation of rows for single-shot multiplexing.
"""
function get_stride_permutation(h_out::Int, w_out::Int, stride::Tuple{Int, Int})
    s_h, s_w = stride
    
    # If stride is 1, no permutation needed
    if s_h == 1 && s_w == 1
        return collect(1:(h_out * w_out))
    end
    
    # Compute the stride pattern based on Orion's single-shot multiplexing
    permutation = Int[]
    
    # Group by stride offset
    for si in 0:(s_h-1)
        for sj in 0:(s_w-1)
            # For each position in the stride pattern (si, sj), gather all outputs at this offset
            for i in 0:((h_out + s_h - 1) ÷ s_h - 1)
                actual_i = i * s_h + si
                if actual_i >= h_out
                    continue
                end
                
                for j in 0:((w_out + s_w - 1) ÷ s_w - 1)
                    actual_j = j * s_w + sj
                    if actual_j >= w_out
                        continue
                    end
                    
                    # Julia uses 1-based indexing
                    push!(permutation, actual_i * w_out + actual_j + 1)
                end
            end
        end
    end
    
    return permutation
end

"""
    build_encoding_plan(input_shape::Tuple{Int, Int, Int, Int}, weight::Array{Float64, 4}, 
                      kernel_size::Tuple{Int, Int}, stride::Tuple{Int, Int}, 
                      padding::Tuple{Int, Int}, dilation::Tuple{Int, Int}, groups::Int)

Builds the full encoding plan for efficient convolution using Orion techniques.

Args:
    input_shape: A tuple (batch, channels, height, width) of the input.
    weight: A 4D array of weights with shape (out_channels, in_channels // groups, kernel_height, kernel_width).
    kernel_size: A tuple (height, width) of the kernel.
    stride: A tuple (height, width) for the stride.
    padding: A tuple (height, width) for the padding.
    dilation: A tuple (height, width) for the dilation.
    groups: An integer specifying the groups for grouped convolution.

Returns:
    A Dict containing the encoding plan with Toeplitz matrices, permutation information, and BSGS metadata.
"""
function build_encoding_plan(input_shape::Tuple{Int, Int, Int, Int}, weight::AbstractArray, 
                          kernel_size::Tuple{Int, Int}, stride::Tuple{Int, Int}, 
                          padding::Tuple{Int, Int}, dilation::Tuple{Int, Int}, groups::Int)
    _, in_channels, h_in, w_in = input_shape
    out_channels, in_channels_per_group, k_h, k_w = size(weight)
    
    # Compute output dimensions
    h_out, w_out = compute_output_shape((h_in, w_in), kernel_size, stride, padding, dilation)
    out_size = h_out * w_out
    in_size = h_in * w_in * (in_channels ÷ groups)
    
    # Step 1: Build the basic Toeplitz matrices for each output channel
    toeplitz_matrices = Vector{Matrix{Float64}}(undef, out_channels)
    
    # Pre-allocate all matrices
    for out_c in 1:out_channels
        toeplitz_matrices[out_c] = zeros(out_size, in_size)
    end
    
    # Precompute position mapping for kernel to input positions
    position_mapping = Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}}()
    for i in 1:h_out, j in 1:w_out, di in 1:k_h, dj in 1:k_w
        in_i = (i-1) * stride[1] + (di-1) * dilation[1] - padding[1] + 1
        in_j = (j-1) * stride[2] + (dj-1) * dilation[2] - padding[2] + 1
        if 1 <= in_i <= h_in && 1 <= in_j <= w_in
            position_mapping[(i, j, di, dj)] = (in_i, in_j)
        end
    end
    
    # Calculate group boundaries
    group_boundaries = [(((out_c-1) ÷ (out_channels ÷ groups)) * (in_channels ÷ groups) + 1,
                         ((out_c-1) ÷ (out_channels ÷ groups) + 1) * (in_channels ÷ groups)) 
                        for out_c in 1:out_channels]
    
    # Parallel matrix construction
    @threads for out_c in 1:out_channels
        in_c_start, in_c_end = group_boundaries[out_c]
        toeplitz = toeplitz_matrices[out_c]
        
        # Parallel fill of matrices
        for ((i, j, di, dj), (in_i, in_j)) in position_mapping
            out_idx = (i-1) * w_out + j
            
            # For each input channel in this group
            for in_c_idx in 1:(in_channels ÷ groups)
                in_c = in_c_start + in_c_idx - 1
                in_idx = (in_c_idx-1) * (h_in * w_in) + (in_i-1) * w_in + in_j
                
                # Get the weight value (adjust for 1-based Julia indexing)
                weight_idx = (in_c_idx, di, dj)
                toeplitz[out_idx, in_idx] = weight[out_c, weight_idx...]
            end
        end
    end
    
    # Step 2: Apply single-shot multiplexing for stride > 1
    # We permute the rows of the Toeplitz matrix based on the stride pattern
    permutation = get_stride_permutation(h_out, w_out, stride)
    permuted_matrices = Vector{Matrix{Float64}}(undef, out_channels)
    
    for out_c in 1:out_channels
        toeplitz = toeplitz_matrices[out_c]
        permuted = zeros(size(toeplitz))
        
        for (new_idx, old_idx) in enumerate(permutation)
            if old_idx <= size(toeplitz, 1)  # Ensure we don't go out of bounds
                permuted[new_idx, :] = toeplitz[old_idx, :]
            end
        end
        
        permuted_matrices[out_c] = permuted
    end
    
    # Step 3: Apply BSGS optimization
    # Divide the matrix into baby-step and giant-step blocks
    bsgs_metadata = Dict{Int, Dict{String, Any}}()
    
    for out_c in 1:out_channels
        toeplitz = permuted_matrices[out_c]
        rows, cols = size(toeplitz)
        
        # Compute optimal factors for BSGS
        n1, n2 = compute_bsgs_factors(rows)
        
        # Create a mapping for efficient BSGS execution
        row_blocks = Tuple{Int, Int}[]
        for block_idx in 0:(n2-1)
            start_row = block_idx * n1 + 1  # 1-based indexing
            end_row = min(start_row + n1 - 1, rows)
            if start_row <= rows  # Only include valid blocks
                push!(row_blocks, (start_row, end_row))
            end
        end
        
        bsgs_metadata[out_c] = Dict(
            "n1" => n1,
            "n2" => n2,
            "row_blocks" => row_blocks
        )
    end
    
    # Create the full encoding plan with all necessary metadata
    encoding_plan = Dict(
        "h_in" => h_in,
        "w_in" => w_in,
        "h_out" => h_out,
        "w_out" => w_out,
        "permutation" => permutation,
        "channels_per_group" => in_channels ÷ groups,
        "permuted_toeplitz" => permuted_matrices,
        "bsgs_metadata" => bsgs_metadata
    )
    
    return encoding_plan
end

"""
    prepare_input_vector(input_sample::Vector{T}, h_in::Int, w_in::Int, channels_per_group::Int, groups::Int) where T <: Union{LWE, PyObject}

Prepares the input data for matrix multiplication by flattening and organizing by groups.

Args:
    input_sample: A vector of LWE ciphertexts representing a single sample from the batch.
    h_in: Input height.
    w_in: Input width.
    channels_per_group: Number of channels per group.
    groups: Number of groups.

Returns:
    A flattened vector of LWE ciphertexts organized for efficient matrix multiplication.
"""
function prepare_input_vector(input_sample::Vector{T}, h_in::Int, w_in::Int, channels_per_group::Int, groups::Int) where T <: Union{LWE, PyObject}
    # Convert PyObjects to LWE if needed
    input_lwes = isa(first(input_sample), PyObject) ? convert_pyobjects_to_lwes(input_sample) : input_sample
    
    # Create the input vector
    input_vector = Vector{LWE}(undef, length(input_lwes))
    
    # For grouped convolutions, organize input by groups
    idx = 1
    for g in 1:groups
        start_idx = (g-1) * channels_per_group * h_in * w_in + 1
        end_idx = g * channels_per_group * h_in * w_in
        
        if end_idx <= length(input_lwes)
            # Extract a copy of the group channels (not a view)
            group_channels = [input_lwes[i] for i in start_idx:end_idx]
            
            # Copy each element
            for el in group_channels
                input_vector[idx] = el
                idx += 1
            end
        end
    end
    
    return input_vector[1:idx-1]  # Return only the filled part
end

# Add a specialized method for SubArray to handle views correctly
function prepare_input_vector(input_sample::SubArray{T, 1}, h_in::Int, w_in::Int, channels_per_group::Int, groups::Int) where T <: Union{LWE, PyObject}
    # Convert the SubArray to a regular Vector
    input_vec = collect(input_sample)
    # Then call the regular method
    return prepare_input_vector(input_vec, h_in, w_in, channels_per_group, groups)
end

"""
    hoisted_dot_product_lwe(input_vector::Vector{T}, row_block::Matrix{Float64}, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}

Applies double-hoisting optimization for multiple dot products.

Args:
    input_vector: Flattened input data as a vector of LWE ciphertexts.
    row_block: A block of Toeplitz matrix rows to process with hoisting.
    zero_ciphertext: A zero LWE ciphertext for initialization.

Returns:
    A vector of results from the dot products.
"""
function hoisted_dot_product_lwe(input_vector::Vector{T}, row_block::AbstractMatrix, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    # In a real FHE implementation, this would use internal hoisting mechanisms
    # Here we simulate by computing each dot product separately, but in parallel
    
    num_rows = size(row_block, 1)
    results = Vector{LWE}(undef, num_rows)
    
    # Pre-extract all rows to avoid view issues
    rows = [Vector{Float64}(row_block[i, :]) for i in 1:num_rows]
    
    # Use threading for sufficiently large blocks
    if num_rows > 8
        @threads for i in 1:num_rows
            results[i] = dot_product(input_vector, rows[i], zero_ciphertext)
        end
    else
        # Sequential for small blocks to avoid thread overhead
        for i in 1:num_rows
            results[i] = dot_product(input_vector, rows[i], zero_ciphertext)
        end
    end
    
    return results
end

"""
    apply_bsgs_forward(input_vector::Vector{T}, encoding_plan::Dict, out_c::Int, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}

Applies BSGS (Baby-Step Giant-Step) algorithm to compute the matrix-vector product efficiently.

Args:
    input_vector: Flattened input data as a vector of LWE ciphertexts.
    encoding_plan: The encoding plan containing Toeplitz matrices and BSGS metadata.
    out_c: The output channel index to process.
    zero_ciphertext: A zero LWE ciphertext for initialization.

Returns:
    The result of the matrix-vector product for the specified output channel.
"""
function apply_bsgs_forward(input_vector::Vector{T}, encoding_plan::Dict, out_c::Int, zero_ciphertext::U) where {T <: Union{LWE, PyObject}, U <: Union{LWE, PyObject}}
    h_out = encoding_plan["h_out"]
    w_out = encoding_plan["w_out"]
    permutation = encoding_plan["permutation"]
    toeplitz = encoding_plan["permuted_toeplitz"][out_c]
    metadata = encoding_plan["bsgs_metadata"][out_c]
    row_blocks = metadata["row_blocks"]
    
    # Output container - initialize with zero ciphertexts
    output = Matrix{LWE}(undef, h_out, w_out)
    
    # Initialize all elements with zero_ciphertext to avoid UndefRefError
    # Use parallel initialization for large matrices
    if h_out * w_out > 100
        @threads for idx in 1:(h_out * w_out)
            i = (idx - 1) ÷ w_out + 1
            j = (idx - 1) % w_out + 1
            output[i, j] = zero_ciphertext
        end
    else
        for i in 1:h_out
            for j in 1:w_out
                output[i, j] = zero_ciphertext
            end
        end
    end
    
    # Step 1: Baby-step rotations and partial products
    partial_results = Vector{Vector{LWE}}(undef, length(row_blocks))
    
    for (block_idx, (start_row, end_row)) in enumerate(row_blocks)
        # Get the block of rows from the Toeplitz matrix
        row_block = toeplitz[start_row:end_row, :]
        
        # Apply hoisted dot product to compute multiple rows efficiently
        partial_results[block_idx] = hoisted_dot_product_lwe(input_vector, row_block, zero_ciphertext)
    end
    
    # Step 2: Giant-step: Combine the partial results into the final output
    result_flat = Vector{LWE}(undef, length(permutation))
    
    # Initialize all elements with zero_ciphertext
    # Use parallel initialization for large arrays
    if length(result_flat) > 100
        @threads for i in 1:length(result_flat)
            result_flat[i] = zero_ciphertext
        end
    else
        for i in 1:length(result_flat)
            result_flat[i] = zero_ciphertext
        end
    end
    
    # Populate the output vector based on the baby-step results
    idx = 1
    for (block_idx, (start_row, end_row)) in enumerate(row_blocks)
        for row_offset in 0:(end_row - start_row)
            flat_idx = start_row + row_offset
            if flat_idx <= length(permutation)
                result_flat[flat_idx] = partial_results[block_idx][row_offset+1]  # +1 for Julia indexing
                idx += 1
            end
        end
    end
    
    # Step 3: De-permute the results to get the correct spatial arrangement
    # This reverses the permutation applied in the single-shot multiplexing
    inverse_permutation = zeros(Int, length(permutation))
    for (new_idx, old_idx) in enumerate(permutation)
        if old_idx <= length(inverse_permutation)
            inverse_permutation[old_idx] = new_idx
        end
    end
    
    # Apply the inverse permutation and reshape to output dimensions
    for i in 1:h_out
        for j in 1:w_out
            out_idx = (i-1) * w_out + j + 1  # +1 for Julia indexing
            if out_idx <= length(permutation)
                perm_idx = inverse_permutation[out_idx]
                if perm_idx > 0 && perm_idx <= length(result_flat)
                    output[i, j] = result_flat[perm_idx]
                end
            end
        end
    end
    
    return output
end

"""
    conv2d_orion_forward(input::Vector{T}, input_shape::Tuple{Int, Int, Int, Int}, 
                      encoding_plan::Dict, bias::Union{Vector{Float64}, Nothing}=nothing) where T <: Union{LWE, PyObject}

Performs the forward pass of a 2D convolution using Orion techniques.

Args:
    input: A vector of LWE ciphertexts representing the flattened input.
    input_shape: A tuple (batch, channels, height, width) of the input.
    encoding_plan: The encoding plan containing Toeplitz matrices and BSGS metadata.
    bias: An optional vector of bias values, one for each output channel.

Returns:
    A vector of LWE ciphertexts representing the flattened output.
"""
function conv2d_orion_forward(input::Vector{T}, input_shape::Tuple{Int, Int, Int, Int}, 
                           encoding_plan::Dict, bias::Union{AbstractVector, Nothing}=nothing) where T <: Union{LWE, PyObject}
    batch_size, in_channels, _, _ = input_shape
    h_out = encoding_plan["h_out"]
    w_out = encoding_plan["w_out"]
    channels_per_group = encoding_plan["channels_per_group"]
    groups = in_channels ÷ channels_per_group
    out_channels = length(encoding_plan["permuted_toeplitz"])
    
    # Convert PyObjects to LWE if needed
    input_lwes = isa(first(input), PyObject) ? convert_pyobjects_to_lwes(input) : input
    
    # Get a zero ciphertext for dot product initialization
    mask_length = length(input_lwes[1].mask)
    zero_ciphertext = LWE(zeros(Float64, mask_length), 0.0)
    
    # Prepare output vector (flattened batch * out_channels * h_out * w_out)
    total_output_size = batch_size * out_channels * h_out * w_out
    output = Vector{LWE}(undef, total_output_size)
    
    # Calculate constants for indexing
    elements_per_sample = in_channels * encoding_plan["h_in"] * encoding_plan["w_in"]
    output_elements_per_batch = out_channels * h_out * w_out
    output_elements_per_channel = h_out * w_out
    
    # Use atomic counters for thread-safe operations
    # Create input vectors for each batch
    input_vectors = Vector{Vector{LWE}}(undef, batch_size)
    
    # First prepare all input vectors (can be done sequentially to avoid view issues)
    for b in 1:batch_size
        sample_start = (b-1) * elements_per_sample + 1
        sample_end = min(b * elements_per_sample, length(input_lwes))
        
        if sample_end >= sample_start
            # Create a full copy rather than a view
            input_sample = [input_lwes[i] for i in sample_start:sample_end]
            input_vectors[b] = prepare_input_vector(
                input_sample, 
                encoding_plan["h_in"], 
                encoding_plan["w_in"], 
                channels_per_group, 
                groups
            )
        end
    end
    
    # Process batches in parallel, with locked channels for better workload balance
    if batch_size > 1
        # Multiple batches - parallelize over batches
        @threads for b in 1:batch_size
            if b <= length(input_vectors) && input_vectors[b] !== nothing
                input_vector = input_vectors[b]
                
                # Process each output channel
                for out_c in 1:out_channels
                    # Apply BSGS forward pass to compute matrix-vector product
                    channel_output = apply_bsgs_forward(input_vector, encoding_plan, out_c, zero_ciphertext)
                    
                    # Add bias if present
                    if bias !== nothing
                        # Apply bias with SIMD optimization when possible
                        for idx in 1:(h_out*w_out)
                            i = (idx-1) ÷ w_out + 1
                            j = (idx-1) % w_out + 1
                            channel_output[i, j] = add(channel_output[i, j], bias[out_c])
                        end
                    end
                    
                    # Store results in the output array (flattened for better cache locality)
                    batch_offset = (b-1) * output_elements_per_batch
                    channel_offset = (out_c-1) * output_elements_per_channel
                    
                    for idx in 1:(h_out*w_out)
                        i = (idx-1) ÷ w_out + 1
                        j = (idx-1) % w_out + 1
                        flat_idx = batch_offset + channel_offset + (i-1)*w_out + j
                        output[flat_idx] = channel_output[i, j]
                    end
                end
            end
        end
    else
        # Single batch - parallelize over channels
        if length(input_vectors) > 0 && input_vectors[1] !== nothing
            input_vector = input_vectors[1]
            
            @threads for out_c in 1:out_channels
                # Apply BSGS forward pass to compute matrix-vector product
                channel_output = apply_bsgs_forward(input_vector, encoding_plan, out_c, zero_ciphertext)
                
                # Add bias if present
                if bias !== nothing
                    for idx in 1:(h_out*w_out)
                        i = (idx-1) ÷ w_out + 1
                        j = (idx-1) % w_out + 1
                        channel_output[i, j] = add(channel_output[i, j], bias[out_c])
                    end
                end
                
                # Store results in the output array
                channel_offset = (out_c-1) * output_elements_per_channel
                
                for idx in 1:(h_out*w_out)
                    i = (idx-1) ÷ w_out + 1
                    j = (idx-1) % w_out + 1
                    flat_idx = channel_offset + (i-1)*w_out + j
                    output[flat_idx] = channel_output[i, j]
                end
            end
        end
    end
    
    return output
end

end