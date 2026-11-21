import numpy as np


def rng_from_seed(seed: int) -> np.random.Generator:
    """Return a NumPy Generator. Pass this around instead of using global RNG state."""
    return np.random.default_rng(seed)
