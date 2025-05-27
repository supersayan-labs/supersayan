module Linear

using Base.Threads
using LinearAlgebra: BLAS

import ...Types: LWE, LWE_ARRAY, LWE_MATRIX, pack_lwe, extract_lwe
import ...Operations: add_lwe, mult_lwe, dot_product_lwe, batch_dot_product_lwe

# Match BLAS threads to Julia threads for maximum throughput
BLAS.set_num_threads(Threads.nthreads())

"""
Forward pass for a linear layer.
"""
function linear_forward(
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

export linear_forward

end
