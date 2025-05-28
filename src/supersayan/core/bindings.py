import os

from juliacall import Main as jl

from supersayan.logging_config import get_logger

logger = get_logger(__name__)

# Load SupersayanTFHE package
current_dir = os.path.dirname(os.path.abspath(__file__))
julia_package_dir = os.path.join(current_dir, "..", "julia_backend")
julia_package_dir = os.path.normpath(julia_package_dir)
logger.info("Loading Julia package from: %s", julia_package_dir)

jl.seval(f'push!(LOAD_PATH, "{julia_package_dir}")')
jl.seval("using SupersayanTFHE")

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False
    logger.info("CuPy not available - CUDA support will be limited")

jl.seval("using DLPack, CUDA, LinearAlgebra, PythonCall")

if HAS_CUPY:
    jl.seval('cupy = pyimport("cupy")')

jl.seval('numpy = pyimport("numpy")')

SupersayanTFHE = jl.SupersayanTFHE
