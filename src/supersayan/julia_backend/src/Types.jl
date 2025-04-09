module Types

export LWE

"""
    struct LWE

An LWE ciphertext consists of a mask (a vector of Float64) and a masked value (Float64).
"""
struct LWE
    mask::Vector{Float64}
    masked::Float64
end

end
