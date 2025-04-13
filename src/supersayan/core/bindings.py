import os
import logging
from julia import Julia

logger = logging.getLogger(__name__)

def initialize_julia():
    """
    Initialize the Julia runtime and import the SupersayanTFHE module.
    
    This function should be called once at startup to ensure the Julia
    backend is properly initialized.
    
    Returns:
        The Julia module for SupersayanTFHE
    """
    try:
        logger.info("Initializing Julia runtime...")
        jl = Julia(compiled_modules=False)
        
        from julia import Main
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        julia_file = os.path.join(current_dir, "..", "julia_backend", "SupersayanTFHE.jl")
        julia_file = os.path.normpath(julia_file)
        
        logger.info("Including Julia backend from: %s", julia_file)
        Main.include(julia_file)
        
        logger.info("Julia backend successfully initialized")
        return Main
    except Exception as e:
        logger.error("Failed to initialize Julia: %s", e)
        raise

# Initialize Julia on module import
try:
    logger.info("Initializing Julia runtime on module import...")
    jl = Julia(compiled_modules=False)
    
    from julia import Main
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    julia_file = os.path.join(current_dir, "..", "julia_backend", "SupersayanTFHE.jl")
    julia_file = os.path.normpath(julia_file)
    
    logger.info("Including Julia backend from: %s", julia_file)
    Main.include(julia_file)
    
    jl = Main
except Exception as e:
    logger.error("Failed to initialize Julia on module import: %s", e)
    logger.warning("You must call initialize_julia() manually before using SuperSayan functions")
    jl = None

__all__ = ["jl", "initialize_julia"]