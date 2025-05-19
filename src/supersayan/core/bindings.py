import os
import logging
from juliacall import Main as jl

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
julia_package_dir = os.path.join(current_dir, "..", "julia_backend")
julia_package_dir = os.path.normpath(julia_package_dir)
logger.info("Loading Julia package from: %s", julia_package_dir)

# FIXME: auto install the julia dependencies
# Refer to: cd /Users/tomjurien/Documents/Projects/SaCS/supersayan/src/supersayan/julia_backend && julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.resolve()'
jl.seval(f'push!(LOAD_PATH, "{julia_package_dir}")')
jl.seval("using SupersayanTFHE")

SupersayanTFHE = jl.SupersayanTFHE