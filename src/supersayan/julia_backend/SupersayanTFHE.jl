module SupersayanTFHE

# Include internal submodules.
include("src/Constants.jl")
include("src/Types.jl")
include("src/Encryption.jl")
include("src/Operations.jl")

# Export the key functionalities.
export Constants, Types, Encryption, Operations

end
