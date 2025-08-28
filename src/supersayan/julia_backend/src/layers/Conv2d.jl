module Conv2d

using Base.Threads: @threads
using LinearAlgebra: BLAS
using CUDA

import ...Types: LWE, LWE_ARRAY, LWE_MATRIX
import ...Operations: add_lwe, mult_lwe

# Match BLAS threads to Julia threads for maximum throughput
BLAS.set_num_threads(Threads.nthreads())

"""
GPU kernel for conv2d forward pass with LWE ciphertexts
"""
function conv2d_forward_kernel!(
    output::CuDeviceArray{Float32,5},
    input::CuDeviceArray{Float32,5},
    weights::CuDeviceArray{Float32,4},
    bias::Union{CuDeviceVector{Float32},Nothing},
    N::Int32,
    C_in::Int32,
    H::Int32,
    W::Int32,
    lwe_dim::Int32,
    C_out::Int32,
    cin_per_g::Int32,
    kh::Int32,
    kw::Int32,
    H_out::Int32,
    W_out::Int32,
    sh::Int32,
    sw::Int32,
    ph::Int32,
    pw::Int32,
    dh::Int32,
    dw::Int32,
    groups::Int32,
)
    # Each thread handles one output position
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total_positions = N * C_out * H_out * W_out

    if idx <= total_positions
        # Decode position
        pos_idx = idx - 1
        w_out = pos_idx % W_out
        pos_idx = pos_idx ÷ W_out
        h_out = pos_idx % H_out
        pos_idx = pos_idx ÷ H_out
        oc = pos_idx % C_out + 1
        n = pos_idx ÷ C_out + 1

        # Group calculations
        g = (oc - 1) ÷ (C_out ÷ groups)
        ic_off = g * cin_per_g

        # Process each LWE dimension
        for lwe_idx = 1:lwe_dim
            acc = 0.0f0

            # Convolution
            for icg = 1:cin_per_g
                for kh_idx = 0:(kh-1)
                    for kw_idx = 0:(kw-1)
                        ih = h_out*sh - ph + kh_idx*dh
                        iw = w_out*sw - pw + kw_idx*dw

                        if 0 ≤ ih < H && 0 ≤ iw < W
                            input_val = input[n, ic_off+icg, ih+1, iw+1, lwe_idx]
                            weight_val = weights[oc, icg, kh_idx+1, kw_idx+1]

                            if lwe_idx == 1  # b component
                                acc += input_val * weight_val
                            else  # a components
                                acc += input_val * weight_val
                            end
                        end
                    end
                end
            end

            # Add bias to b component only
            if lwe_idx == 1 && bias !== nothing
                acc += bias[oc]
            end

            output[n, oc, h_out+1, w_out+1, lwe_idx] = acc
        end
    end

    return nothing
end

"""
CPU version of conv2d forward pass
"""
function conv2d_forward_cpu(
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
    @threads for idx = 1:(N*C_out)
        n = (idx - 1) ÷ C_out + 1
        oc = (idx - 1) % C_out + 1

        g = (oc - 1) ÷ (C_out ÷ groups)
        ic_off = g * cin_per_g

        @inbounds for oh = 0:(H_out-1), ow = 0:(W_out-1)
            acc = zero_cipher

            for icg = 1:cin_per_g, kh_idx = 0:(kh-1), kw_idx = 0:(kw-1)
                ih = oh*sh - ph + kh_idx*dh
                iw = ow*sw - pw + kw_idx*dw
                if 0 ≤ ih < H && 0 ≤ iw < W
                    enc = input[n, ic_off+icg, ih+1, iw+1, :]
                    wval = weights[oc, icg, kh_idx+1, kw_idx+1]
                    acc = add_lwe(acc, mult_lwe(enc, wval))
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

"""
GPU version of conv2d forward pass
"""
function conv2d_forward_gpu(
    input::CuArray{Float32,5},
    weights::CuArray{Float32,4},
    bias::Union{CuArray{Float32,1},Nothing} = nothing,
    stride::Tuple{Int,Int} = (1, 1),
    padding::Tuple{Int,Int} = (0, 0),
    dilation::Tuple{Int,Int} = (1, 1),
    groups::Int = 1,
)::CuArray{Float32,5}
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

    # Allocate output on GPU
    output = CUDA.zeros(Float32, N, C_out, H_out, W_out, lwe_dim)

    # Launch kernel
    total_positions = N * C_out * H_out * W_out
    threads = 256
    blocks = cld(total_positions, threads)

    @cuda threads=threads blocks=blocks conv2d_forward_kernel!(
        output,
        input,
        weights,
        bias,
        Int32(N),
        Int32(C_in),
        Int32(H),
        Int32(W),
        Int32(lwe_dim),
        Int32(C_out),
        Int32(cin_per_g),
        Int32(kh),
        Int32(kw),
        Int32(H_out),
        Int32(W_out),
        Int32(sh),
        Int32(sw),
        Int32(ph),
        Int32(pw),
        Int32(dh),
        Int32(dw),
        Int32(groups),
    )

    return output
end

"""
Forward pass for a convolutional layer.
Automatically dispatches to CPU or GPU implementation based on input type.
"""
function conv2d_forward(
    input::Union{LWE_MATRIX,CuArray{Float32,5}},
    weights::Union{AbstractArray{Float32,4},CuArray{Float32,4}},
    bias::Union{AbstractVector{Float32},CuArray{Float32,1},Nothing} = nothing,
    stride::Tuple{Int,Int} = (1, 1),
    padding::Tuple{Int,Int} = (0, 0),
    dilation::Tuple{Int,Int} = (1, 1),
    groups::Int = 1,
)::Union{AbstractArray{Float32,5},CuArray{Float32,5}}
    # Ensure all inputs are on the same device
    if isa(input, CuArray)
        weights = isa(weights, CuArray) ? weights : CuArray(weights)
        bias = bias === nothing ? nothing : (isa(bias, CuArray) ? bias : CuArray(bias))
        return conv2d_forward_gpu(input, weights, bias, stride, padding, dilation, groups)
    else
        weights = isa(weights, CuArray) ? Array(weights) : weights
        bias = bias === nothing ? nothing : (isa(bias, CuArray) ? Array(bias) : bias)
        return conv2d_forward_cpu(input, weights, bias, stride, padding, dilation, groups)
    end
end

export conv2d_forward

end
