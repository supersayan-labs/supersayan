module Linear

using Base.Threads
using LinearAlgebra: BLAS

import ...Types: LWE, LWE_ARRAY, LWE_BATCH, pack_lwe, extract_lwe
import ...Operations: add_lwe, mult_lwe, dot_product_lwe, batch_dot_product_lwe

# Match BLAS threads to Julia threads
BLAS.set_num_threads(Threads.nthreads())

"""
Performs a linear transformation on a batch of LWE ciphertexts.
"""
function linear_forward(
    input::LWE_BATCH,
    weights::AbstractMatrix{Float32},
    bias::Union{AbstractVector{Float32},Nothing} = nothing,
)::LWE_BATCH
    batch_size, in_features, lwe_dim = size(input)
    out_features = size(weights, 1)

    @assert size(weights, 2) == in_features "Weight columns must match input features"

    if bias !== nothing
        @assert length(bias) == out_features "Bias length must match out_features"
    end

    output = Array{Float32,3}(undef, batch_size, out_features, lwe_dim)

    zero_cipher = zeros(Float32, lwe_dim)

    for b = 1:batch_size
        input_b = @view input[b, :, :]

        for j = 1:out_features
            weight_row = @view weights[j, :]
            result = dot_product_lwe(input_b, weight_row, zero_cipher)

            if bias !== nothing
                result = add_lwe(result, bias[j])
            end

            output[b, j, :] = result
        end
    end

    return output
end

export linear_forward

end
