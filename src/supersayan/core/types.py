import numpy as np
from typing import Tuple

A = np.ndarray[np.float32]
B = np.float32
LWE = np.ndarray[np.float32]
MU = np.float32
SIGMA = np.float32
KEY = np.ndarray[np.float32]
P = np.int32

def extract_lwe(x: LWE) -> Tuple[A, B]:
    return x[1:], x[0]

def pack_lwe(x: A, y: B) -> LWE:
    return np.concatenate(([y], x))