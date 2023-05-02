"""
Example Experimentalist Sampler
"""
import numpy as np

def example_sampler(
    X: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Include inline mathematics in docstring \\(x < 1\\) or $c = 3$
    or block mathematics:

    \\[
        x + 1 = 3
    \\]


    $$
    y + 1 = 4
    $$

    """

    return X[:n]

