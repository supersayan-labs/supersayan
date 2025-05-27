module Conv2d

import ...Types: LWE, LWE_ARRAY, LWE_MATRIX
import ...Operations: add_lwe, mult_lwe

using Base.Threads: @threads
using LinearAlgebra: BLAS
BLAS.set_num_threads(Threads.nthreads())


function conv2d_forward(
    input::LWE_MATRIX,
    weights::AbstractArray{Float32,4},
    bias::Union{AbstractVector{Float32},Nothing} = nothing,
    stride::Tuple{Int,Int} = (1, 1),
    padding::Tuple{Int,Int} = (0, 0),
    dilation::Tuple{Int,Int} = (1, 1),
    groups::Int = 1,
)::AbstractArray{Float32,5}
    # Dimension bookkeeping
    N, C_in, H, W, lwe_dim = size(input)
    C_out, cin_per_g, kh, kw = size(weights)

    @assert C_in % groups == 0 "C_in must be divisible by groups"
    @assert C_out % groups == 0 "C_out must be divisible by groups"
    @assert cin_per_g == C_in ÷ groups "weights dim 2 must be C_in/groups"

    cout_per_g = C_out ÷ groups
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    kh_eff = dh*(kh-1) + 1
    kw_eff = dw*(kw-1) + 1
    H_out = (H + 2ph - kh_eff) ÷ sh + 1
    W_out = (W + 2pw - kw_eff) ÷ sw + 1

    # Buffers
    output = Array{Float32,5}(undef, N, C_out, H_out, W_out, lwe_dim)
    zero_cipher = zeros(Float32, lwe_dim)

    # Convolution loop
    @threads for idx in 1:(N*C_out)
        n  = (idx - 1) ÷ C_out + 1
        oc = (idx - 1) % C_out + 1

        g      = (oc - 1) ÷ (C_out ÷ groups)
        ic_off = g * cin_per_g

        @inbounds for oh = 0:(H_out-1), ow = 0:(W_out-1)
            acc = zero_cipher

            for icg = 1:cin_per_g, kh_idx = 0:(kh-1), kw_idx = 0:(kw-1)
                ih = oh*sh - ph + kh_idx*dh
                iw = ow*sw - pw + kw_idx*dw
                if 0 ≤ ih < H && 0 ≤ iw < W
                    enc  = input[n, ic_off+icg, ih+1, iw+1, :]
                    wval = weights[oc, icg, kh_idx+1, kw_idx+1]
                    acc  = add_lwe(acc, mult_lwe(enc, wval))
                end
            end

            # fuse bias addition here
            if bias !== nothing
                acc = add_lwe(acc, bias[oc])
            end

            output[n, oc, oh+1, ow+1, :] = acc
        end
    end

    return output
end

export conv2d_forward

end
