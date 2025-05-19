# Conv2d.jl
module Conv2d

using PyCall
using Base.Threads
using LinearAlgebra: BLAS

# Import from parent module
import SupersayanTFHE.Types: LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes
import SupersayanTFHE.Operations: add, mult, dot_product

export conv2d_forward

"""
    conv2d_forward(cipher_vec::Vector{T},
                   input_shape::Tuple{Int,Int,Int,Int},
                   weight::Array{<:Real,4},
                   ksize::Tuple{Int,Int},
                   stride::Tuple{Int,Int},
                   padding::Tuple{Int,Int},
                   dilation::Tuple{Int,Int},
                   groups::Int,
                   bias::Union{Vector{<:Real},Nothing}=nothing) where T <: Union{LWE,PyObject}

Perform a "naïve" sliding-window 2D convolution on LWE ciphertexts:

- `cipher_vec` is a flat Vector of length N*C_in*H*W (row-major).
- `weight` is plain-text Float kernel of size (C_out, C_in/groups, kh, kw).
- `bias` if given is a plain-text vector of length C_out.

Uses `dot_product` to multiply each window by the plain kernel slice, and `add` to accumulate.
Parallel over batch dimension.
"""
function conv2d_forward(
    cipher_vec::Vector{T},
    input_shape::Tuple{Int,Int,Int,Int},
    weight::Array{<:Real,4},
    ksize::Tuple{Int,Int},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
    dilation::Tuple{Int,Int},
    groups::Int,
    bias::Union{Vector{<:Real},Nothing}=nothing,
) where T <: Union{LWE,PyObject}

    # Unpack shapes
    N, C_in, H, W = input_shape
    kh, kw       = ksize
    sh, sw       = stride
    ph, pw       = padding
    dh, dw       = dilation
    C_out        = size(weight, 1)
    @assert C_in % groups == 0 "in_channels must be divisible by groups"
    @assert C_out % groups == 0 "out_channels must be divisible by groups"
    g_in  = C_in  ÷ groups
    g_out = C_out ÷ groups

    # Convert PyObjects if needed
    input = isa(cipher_vec[1], PyObject) ?
        convert_pyobjects_to_lwes(cipher_vec) : cipher_vec

    # Compute output H_out, W_out
    H_out = (H + 2ph - dh*(kh-1) - 1) ÷ sh + 1
    W_out = (W + 2pw - dw*(kw-1) - 1) ÷ sw + 1

    total_out = N * C_out * H_out * W_out
    output = Vector{LWE}(undef, total_out)

    # Pre-allocate a zero ciphertext for bias if needed
    zero_ct = LWE(zeros(eltype(input[1].mask), length(input[1].mask)), 0.0)

    @threads for n in 1:N
        # Offsets into flat vectors
        in_off  = (n-1)*C_in*H*W
        out_off = (n-1)*C_out*H_out*W_out

        for g in 1:groups
            # slice of weight for this group: shape (g_out, g_in, kh, kw)
            wgrp = view(weight, (g-1)*g_out+1:g*g_out, :, :, :)

            for oc in 1:g_out
                # flatten the kh×kw×g_in kernel into one vector
                kw_flat = vec(@view wgrp[oc, :, :, :])

                # optional bias for this out‐channel
                bval = bias === nothing ? nothing : bias[(g-1)*g_out + oc]

                for oh in 0:H_out-1, ow in 0:W_out-1
                    # gather one window across all g_in channels
                    window = Vector{LWE}(undef, g_in*kh*kw)
                    idx = 1
                    for ic in 0:g_in-1
                        base = in_off + (g-1)*g_in*H*W + ic*H*W
                        for r in 0:kh-1, c in 0:kw-1
                            ih = oh*sh + r*dh - ph
                            iw = ow*sw + c*dw - pw
                            if 0 ≤ ih < H && 0 ≤ iw < W
                                window[idx] = input[base + ih*W + iw + 1]
                            else
                                # explicit zero-padding
                                window[idx] = zero_ct
                            end
                            idx += 1
                        end
                    end

                    # encrypted dot with plaintext kernel
                    sum_ct = dot_product(window, kw_flat, zero_ct)

                    # add bias if given
                    if bval !== nothing
                        sum_ct = add(sum_ct, bval)
                    end

                    # store result
                    out_idx = out_off +
                              (g-1)*g_out*H_out*W_out +
                              (oc-1)*H_out*W_out +
                              oh*W_out + ow + 1
                    output[out_idx] = sum_ct
                end
            end
        end
    end

    return output
end

end # module