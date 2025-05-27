module Types

using PythonCall

const A = AbstractArray{Float32}
const B = Float32
const LWE = AbstractArray{Float32}
const LWE_ARRAY = AbstractArray{Float32,2}
const MU = Float32
const SIGMA = Float32
const KEY = AbstractArray{Float32}
const P = Int32
const LWE_BATCH = AbstractArray{Float32,3}
const LWE_MATRIX = AbstractArray{Float32}

"""
    extract_lwe(x::LWE)::Tuple{A, B}

Extract mask and masked value from an LWE ciphertext.
Returns tuple of (mask, masked value).
"""
function extract_lwe(x::LWE)::Tuple{A,B}
    return x[2:end], x[1]
end

"""
    pack_lwe(x::A, y::B)::LWE

Pack mask and masked value into an LWE ciphertext.
Returns the combined LWE vector.
"""
function pack_lwe(x::A, y::B)::LWE
    return [y; x]
end

export A, B, LWE, MU, SIGMA, KEY, P, extract_lwe, pack_lwe
export LWE_ARRAY, LWE_BATCH

end
