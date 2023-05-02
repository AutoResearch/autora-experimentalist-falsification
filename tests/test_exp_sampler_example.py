from autora.experimentalist.sampler.example import example_sampler
import numpy as np

# Note: We encourage you to write more functionality tests for your sampler.

def test_output_dimensions():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    n = 2
    X_new = example_sampler(X, n)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])
