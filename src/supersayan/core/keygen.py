from .bindings import SupersayanTFHE
from .types import KEY


def generate_secret_key() -> KEY:
    """
    Generate a secret LWE key using the Julia backend.
    Returns the key as a Python list.

    Returns:
        KEY: The secret key
    """
    return SupersayanTFHE.Encryption.generate_key()
