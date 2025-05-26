module Conv2d

using Base.Threads
using LinearAlgebra: BLAS

import ...Types: LWE, LWE_ARRAY, LWE_BATCH, pack_lwe, extract_lwe
import ...Operations: add_lwe, mult_lwe, dot_product_lwe, conv2d_lwe

"""
Performs a 2D convolution on a batch of LWE ciphertexts.
"""
function conv2d_forward(
    input::LWE_BATCH,
    weights::Array{Float32,4},
    bias::Union{Vector{Float32},Nothing} = nothing,
    stride::Tuple{Int,Int} = (1, 1),
    padding::Tuple{Int,Int} = (0, 0),
)::LWE_BATCH
    lwe_dim = size(input, 4)
    zero_cipher = zeros(Float32, lwe_dim)

    output = conv2d_lwe(input, weights, stride, padding, zero_cipher)

    if bias !== nothing
        batch_size, out_channels, spatial_size, _ = size(output)

        for b = 1:batch_size
            for oc = 1:out_channels
                for s = 1:spatial_size
                    output[b, oc, s, :] = add_lwe(@view(output[b, oc, s, :]), bias[oc])
                end
            end
        end
    end

    return output
end

export conv2d_forward

end
