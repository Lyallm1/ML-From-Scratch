import numpy as np

def linear_kernel(**kwargs):
    return lambda x1, x2: np.inner(x1, x2)
def polynomial_kernel(power, coef, **kwargs):
    return lambda x1, x2: (np.inner(x1, x2) + coef)**power
def rbf_kernel(gamma, **kwargs):
    return lambda x1, x2: np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
