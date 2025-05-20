from .bindings import SupersayanTFHE
from .types import KEY


def generate_secret_key() -> KEY:
    """
    Generate a secret LWE key using the Julia backend.
    Returns the key as a Python list.

    Returns:
        list[int]: The secret key as a list of integers
    """
    return SupersayanTFHE.Encryption.generate_key()