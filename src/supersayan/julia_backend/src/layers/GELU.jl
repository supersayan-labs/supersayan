module GELU

using Base.Threads
using LinearAlgebra: BLAS
using CUDA

import ...Types: LWE, LWE_ARRAY, LWE_MATRIX, pack_lwe, extract_lwe
import ...Operations: add_lwe, mult_lwe

# Match BLAS threads to Julia threads for maximum throughput
BLAS.set_num_threads(Threads.nthreads())

"""
Polynomial approximation coefficients for GELU using tanh approximation:
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

For FHE implementation, we use a polynomial approximation of tanh and simplify:
GELU(x) ≈ 0.5 * x * (1 + approx_tanh(0.7978845608 * (x + 0.044715 * x^3)))

Where approx_tanh(z) ≈ z - z^3/3 + 2*z^5/15 for |z| < 1
"""
const GELU_COEFF_0 = 0.5f0
const GELU_COEFF_1 = 0.7978845608f0  # sqrt(2/π)
const GELU_COEFF_2 = 0.044715f0

"""
GPU kernel for GELU activation with LWE ciphertexts using polynomial approximation
"""
function gelu_forward_kernel!(
    output::CuDeviceArray{Float32},
    input::CuDeviceArray{Float32},
    total_elements::Int32,
    lwe_dim::Int32,
    approximate::Bool,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= total_elements
        # Decode the multi-dimensional index
        flat_idx = idx - 1
        lwe_idx = (flat_idx % lwe_dim) + 1
        element_idx = flat_idx ÷ lwe_dim + 1

        # Calculate how many complete elements we've processed
        elements_per_thread = total_elements ÷ lwe_dim

        if element_idx <= elements_per_thread
            # Calculate the actual position in the flattened input
            input_pos = (element_idx - 1) * lwe_dim + lwe_idx

            if input_pos <= length(input)
                x = input[input_pos]

                if approximate
                    # Polynomial approximation: x * (0.5 + 0.5 * tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                    # Simplified polynomial for FHE: 0.5*x + c1*x^3 + c2*x^5
                    x3 = x * x * x
                    x5 = x3 * x * x

                    result = 0.5f0 * x + 0.3989f0 * x3 - 0.1063f0 * x5
                else
                    # Simpler linear approximation for faster computation
                    result = 0.5f0 * x * (1.0f0 + 0.7978845608f0 * x)
                end

                output[input_pos] = result
            end
        end
    end

    return nothing
end

"""
CPU version of GELU forward pass with LWE encryption
"""
function gelu_forward_cpu(input::LWE_MATRIX, approximate::String = "none")::LWE_MATRIX
    input_shape = size(input)
    total_dims = length(input_shape)

    # Handle different input shapes (3D, 4D, 5D tensors)
    if total_dims == 3
        batch_size, features, lwe_dim = input_shape
        output = Array{Float32,3}(undef, batch_size, features, lwe_dim)

        @threads for b = 1:batch_size
            for f = 1:features
                x_encrypted = input[b, f, :]
                output[b, f, :] = gelu_activation_lwe(x_encrypted, approximate)
            end
        end
    elseif total_dims == 5
        batch_size, channels, height, width, lwe_dim = input_shape
        output = Array{Float32,5}(undef, batch_size, channels, height, width, lwe_dim)

        @threads for b = 1:batch_size
            for c = 1:channels
                for h = 1:height
                    for w = 1:width
                        x_encrypted = input[b, c, h, w, :]
                        output[b, c, h, w, :] =
                            gelu_activation_lwe(x_encrypted, approximate)
                    end
                end
            end
        end
    else
        throw(ArgumentError("Unsupported input shape: $input_shape"))
    end

    return output
end

"""
Apply GELU activation to a single LWE-encrypted value
"""
function gelu_activation_lwe(
    x_encrypted::AbstractVector{Float32},
    approximate::String,
)::AbstractVector{Float32}
    if approximate == "tanh"
        # Polynomial approximation: GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x^3)))
        # Simplified for FHE: 0.5*x + c1*x^3 + c2*x^5

        x = x_encrypted
        x3 = mult_lwe(mult_lwe(x, x), x)  # x^3
        x5 = mult_lwe(mult_lwe(x3, x), x)  # x^5

        term1 = mult_lwe(x, 0.5f0)
        term2 = mult_lwe(x3, 0.3989f0)
        term3 = mult_lwe(x5, -0.1063f0)

        result = add_lwe(add_lwe(term1, term2), term3)
        return result
    else
        # Linear approximation: GELU(x) ≈ 0.5*x*(1 + 0.7978845608*x)
        x = x_encrypted
        x_scaled = mult_lwe(x, 0.7978845608f0)
        one_plus_scaled = add_lwe(x_scaled, 1.0f0)
        result = mult_lwe(mult_lwe(x, one_plus_scaled), 0.5f0)
        return result
    end
end

"""
GPU version of GELU forward pass
"""
function gelu_forward_gpu(
    input::CuArray{Float32},
    approximate::String = "none",
)::CuArray{Float32}
    input_shape = size(input)
    total_elements = length(input)
    lwe_dim = input_shape[end]  # Last dimension is always LWE dimension

    # Allocate output on GPU
    output = CUDA.zeros(Float32, input_shape...)

    # Launch kernel
    threads = 256
    blocks = cld(total_elements, threads)

    use_approx = (approximate == "tanh")

    @cuda threads=threads blocks=blocks gelu_forward_kernel!(
        output,
        input,
        Int32(total_elements),
        Int32(lwe_dim),
        use_approx,
    )

    return output
end

"""
Forward pass for GELU activation.
Automatically dispatches to CPU or GPU implementation based on input type.
"""
function gelu_forward(
    input::Union{LWE_MATRIX,CuArray{Float32}},
    approximate::String = "none",
)::Union{LWE_MATRIX,CuArray{Float32}}
    if isa(input, CuArray)
        return gelu_forward_gpu(input, approximate)
    else
        return gelu_forward_cpu(input, approximate)
    end
end

export gelu_forward

end
