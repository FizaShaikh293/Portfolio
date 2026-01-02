# src/privacy_utils.py
# Small utilities for differential privacy. We keep this intentionally simple so it is easy to inspect.

import numpy as np

def laplace_mechanism(value: float, sensitivity: float, epsilon: float) -> float:
    """
    Add Laplace noise proportional to sensitivity / epsilon.

    Note:
        This is a basic implementation intended for demonstration and teaching.
        In production, always rely on a vetted differential privacy library.
    """
    scale = sensitivity / float(epsilon)
    noise = np.random.laplace(loc=0.0, scale=scale)
    return float(value + noise)


def apply_dp_to_series(values, sensitivity, epsilon):
    """
    Apply the Laplace mechanism element-wise to a list or pandas Series.

    Args:
        values: iterable of numbers
        sensitivity: sensitivity of the query
        epsilon: privacy budget

    Returns:
        List of noisy values
    """
    return [laplace_mechanism(float(v), sensitivity, epsilon) for v in values]
