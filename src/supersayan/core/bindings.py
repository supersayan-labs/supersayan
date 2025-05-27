import os
from juliacall import Main as jl
from supersayan.logging_config import get_logger

logger = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
julia_package_dir = os.path.join(current_dir, "..", "julia_backend")
julia_package_dir = os.path.normpath(julia_package_dir)
logger.info("Loading Julia package from: %s", julia_package_dir)

jl.seval(f'push!(LOAD_PATH, "{julia_package_dir}")')
jl.seval("using SupersayanTFHE")

SupersayanTFHE = jl.SupersayanTFHE
