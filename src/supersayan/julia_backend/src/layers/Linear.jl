module Linear

using Base.Threads
using LinearAlgebra: BLAS
using CUDA

import ...Types: LWE, LWE_ARRAY, LWE_MATRIX, pack_lwe, extract_lwe
import ...Operations: add_lwe, mult_lwe, dot_product_lwe, batch_dot_product_lwe

# Match BLAS threads to Julia threads for maximum throughput
BLAS.set_num_threads(Threads.nthreads())

"""
GPU kernel for linear layer forward pass with LWE ciphertexts
"""
function linear_forward_kernel!(
    output::CuDeviceArray{Float32,3},
    input::CuDeviceArray{Float32,3},
    weights::CuDeviceMatrix{Float32},
    bias::Union{CuDeviceVector{Float32},Nothing},
    batch_size::Int32,
    in_features::Int32,
    out_features::Int32,
    lwe_dim::Int32,
)
    # Each thread handles one output element (batch, out_feature, lwe_component)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total_elements = batch_size * out_features * lwe_dim

    if idx <= total_elements
        # Decode the 3D index
        lwe_idx = (idx - 1) % lwe_dim + 1
        temp = (idx - 1) ÷ lwe_dim
        out_idx = temp % out_features + 1
        batch_idx = temp ÷ out_features + 1

        # Accumulate the dot product for this output
        acc = 0.0f0

        # Handle 'a' components (lwe_idx > 1)
        if lwe_idx > 1
            for i = 1:in_features
                # input[batch_idx, i, lwe_idx] * weights[out_idx, i]
                acc += input[batch_idx, i, lwe_idx] * weights[out_idx, i]
            end
        else  # Handle 'b' component (lwe_idx == 1)
            for i = 1:in_features
                # For b component: multiply input's b by weight
                acc += input[batch_idx, i, 1] * weights[out_idx, i]
            end

            # Add bias if present
            if bias !== nothing
                acc += bias[out_idx]
            end
        end

        output[batch_idx, out_idx, lwe_idx] = acc
    end

    return nothing
end

"""
CPU version of linear forward pass
"""
function linear_forward_cpu(
    input::LWE_MATRIX,
    weights::AbstractMatrix{Float32},
    bias::Union{AbstractVector{Float32},Nothing} = nothing,
)::LWE_MATRIX
    batch_size, in_features, lwe_dim = size(input)
    out_features = size(weights, 1)

    @assert size(weights, 2) == in_features "Weight columns must match input features"

    if bias !== nothing
        @assert length(bias) == out_features "Bias length must match out_features"
    end

    output = Array{Float32,3}(undef, batch_size, out_features, lwe_dim)

    zero_cipher = zeros(Float32, lwe_dim)

    # parallelize over batch examples
    @threads for b = 1:batch_size
        input_b = view(input,b,:,:)

        for j = 1:out_features
            res = dot_product_lwe(input_b, view(weights, j, :), zero_cipher)
            if bias !== nothing
                res = add_lwe(res, bias[j])
            end
            output[b, j, :] = res
        end
    end

    return output
end

"""
GPU version of linear forward pass
"""
function linear_forward_gpu(
    input::CuArray{Float32,3},
    weights::CuArray{Float32,2},
    bias::Union{CuArray{Float32,1},Nothing} = nothing,
)::CuArray{Float32,3}
    batch_size, in_features, lwe_dim = size(input)
    out_features = size(weights, 1)

    @assert size(weights, 2) == in_features "Weight columns must match input features"

    if bias !== nothing
        @assert length(bias) == out_features "Bias length must match out_features"
    end

    # Allocate output on GPU
    output = CUDA.zeros(Float32, batch_size, out_features, lwe_dim)

    # Launch kernel
    total_elements = batch_size * out_features * lwe_dim
    threads = 256
    blocks = cld(total_elements, threads)

    @cuda threads=threads blocks=blocks linear_forward_kernel!(
        output,
        input,
        weights,
        bias,
        Int32(batch_size),
        Int32(in_features),
        Int32(out_features),
        Int32(lwe_dim),
    )

    return output
end

"""
Forward pass for a linear layer.
Automatically dispatches to CPU or GPU implementation based on input type.
"""
function linear_forward(
    input::Union{LWE_MATRIX,CuArray{Float32,3}},
    weights::Union{AbstractMatrix{Float32},CuArray{Float32,2}},
    bias::Union{AbstractVector{Float32},CuArray{Float32,1},Nothing} = nothing,
)::Union{LWE_MATRIX,CuArray{Float32,3}}
    # Ensure all inputs are on the same device
    if isa(input, CuArray)
        weights = isa(weights, CuArray) ? weights : CuArray(weights)
        bias = bias === nothing ? nothing : (isa(bias, CuArray) ? bias : CuArray(bias))
        return linear_forward_gpu(input, weights, bias)
    else
        weights = isa(weights, CuArray) ? Array(weights) : weights
        bias = bias === nothing ? nothing : (isa(bias, CuArray) ? Array(bias) : bias)
        return linear_forward_cpu(input, weights, bias)
    end
end

export linear_forward

end
