module Conv2d

using Base.Threads
using LinearAlgebra: BLAS

import ...Types: LWE, LWE_ARRAY, LWE_BATCH, pack_lwe, extract_lwe
import ...Operations: add_lwe, mult_lwe, dot_product_lwe

"""
Performs a 2D convolution on a batch of LWE ciphertexts.
"""
function conv2d_forward(
    input::LWE_BATCH,
    weights::Array{Float32,4},
    ksize::Tuple{Int,Int},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
    dilation::Tuple{Int,Int},
    groups::Int,
    bias::Union{Vector{Float32},Nothing} = nothing,
)::LWE_BATCH
    # Input shape: (batch_size, channels_in, height * width, lwe_dim)
    batch_size, channels_in, spatial_size, lwe_dim = size(input)

    # Deduce H and W from spatial_size
    # Assuming square input for simplicity, adjust if needed
    H = W = Int(sqrt(spatial_size))
    @assert H * W == spatial_size "Input spatial dimensions must be square"

    # Weight shape: (channels_out, channels_in/groups, kh, kw)
    channels_out, channels_per_group, kh, kw = size(weights)

    @assert channels_in % groups == 0 "in_channels must be divisible by groups"
    @assert channels_out % groups == 0 "out_channels must be divisible by groups"
    @assert channels_per_group == channels_in ÷ groups "Weight shape mismatch"

    # Unpack convolution parameters
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Calculate output dimensions
    H_out = (H + 2ph - dh*(kh-1) - 1) ÷ sh + 1
    W_out = (W + 2pw - dw*(kw-1) - 1) ÷ sw + 1
    spatial_out = H_out * W_out

    # Initialize output
    output = Array{Float32,4}(undef, batch_size, channels_out, spatial_out, lwe_dim)

    # Pre-allocate zero ciphertext
    zero_cipher = zeros(Float32, lwe_dim)

    # Process each batch
    for b = 1:batch_size
        # Process each group
        for g = 1:groups
            # Channel ranges for this group
            in_start = (g-1) * channels_per_group + 1
            in_end = g * channels_per_group
            out_start = (g-1) * (channels_out ÷ groups) + 1
            out_end = g * (channels_out ÷ groups)

            # Process each output channel in this group
            for oc = out_start:out_end
                # Get weights for this output channel
                weight_slice = @view weights[oc, :, :, :]

                # Process each output spatial position
                for oh = 0:(H_out-1), ow = 0:(W_out-1)
                    # Gather input window
                    window_size = channels_per_group * kh * kw
                    window = Matrix{Float32}(undef, window_size, lwe_dim)
                    weight_flat = Vector{Float32}(undef, window_size)

                    idx = 1
                    for ic = 1:channels_per_group
                        for r = 0:(kh-1), c = 0:(kw-1)
                            ih = oh*sh + r*dh - ph
                            iw = ow*sw + c*dw - pw

                            if 0 ≤ ih < H && 0 ≤ iw < W
                                # Map to flattened spatial index
                                spatial_idx = ih * W + iw + 1
                                window[idx, :] =
                                    @view input[b, in_start+ic-1, spatial_idx, :]
                            else
                                # Zero padding
                                window[idx, :] = zero_cipher
                            end

                            weight_flat[idx] = weight_slice[ic, r+1, c+1]
                            idx += 1
                        end
                    end

                    # Compute dot product
                    result = dot_product_lwe(window, weight_flat, zero_cipher)

                    # Add bias if provided
                    if bias !== nothing
                        result = add_lwe(result, bias[oc])
                    end

                    # Store result
                    out_spatial_idx = oh * W_out + ow + 1
                    output[b, oc, out_spatial_idx, :] = result
                end
            end
        end
    end

    return output
end

export conv2d_forward

end
