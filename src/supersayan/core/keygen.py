from .bindings import SupersayanTFHE


def generate_secret_key() -> list[int]:
    """
    Generate a secret LWE key using the Julia backend.
    Returns the key as a Python list.

    Returns:
        list[int]: The secret key as a list of integers
    """
    key = SupersayanTFHE.Encryption.generate_key()

    return list(key)
