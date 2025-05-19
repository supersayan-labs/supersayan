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