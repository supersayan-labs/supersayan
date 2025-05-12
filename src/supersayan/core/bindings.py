import os
import logging
from julia import Julia, Main

logger = logging.getLogger(__name__)

try:
    # Initialize Julia runtime
    logger.info("Initializing Julia runtime")
    Julia(compiled_modules=False, threads=True)
    logger.info("Julia runtime initialized")
except Exception as e:
    logger.error("Failed to initialize Julia: %s", e)
    raise

current_dir = os.path.dirname(os.path.abspath(__file__))
julia_file = os.path.join(current_dir, "..", "julia_backend", "SupersayanTFHE.jl")
julia_file = os.path.normpath(julia_file)
logger.info("Including Julia backend from: %s", julia_file)
Main.include(julia_file)

# Get Julia module in Main namespace
SupersayanTFHE = Main.SupersayanTFHE

__all__ = ["SupersayanTFHE"]


def jlwrap_to_bytes(jlw):
    """
    Convert a PyCall.jlwrap into plain Python `bytes` using Julia's
    Base.serialize.  The companion function on the *client* will turn those
    bytes back into a Julia object.
    """
    # Define a Julia helper only once
    if not hasattr(jlwrap_to_bytes, "_defined"):
        Main.eval(
            """
            function _py_serialize_to_bytes(x)
                io = IOBuffer()
                serialize(io, x)
                take!(io)     # Vector{UInt8}
            end
            """
        )
        _jlwrap_to_bytes._defined = True  # type: ignore[attr-defined]
    vec_u8 = Main._py_serialize_to_bytes(jlw)
    return bytes(vec_u8)

def bytes_to_jlwrap(buf: bytes):
    """
    Convert the `bytes` obtained from `_jlwrap_to_bytes()` on the server
    back into a `PyCall.jlwrap` object.

    >>> jl_obj = some_julia_call()       # PyCall.jlwrap
    >>> blob   = _jlwrap_to_bytes(jl_obj)
    >>> roundtrip = bytes_to_jlwrap(blob)
    """
    # Define the Julia helper only once (idempotent)
    if not hasattr(bytes_to_jlwrap, "_defined"):
        Main.eval(
            """
            _py_deserialize_from_bytes(b) = deserialize(IOBuffer(b))
            """
        )
        bytes_to_jlwrap._defined = True

    return Main._py_deserialize_from_bytes(buf)