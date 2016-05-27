import numpy as np

def l2norm(array1,array2):
    """Calculate sqrt ( sum |array1 - array2|^2 / sum|array1|^2 )."""
    tot  = np.sum((array1 * array1.conj()).real**2)
    diff = array1-array2
    return np.sqrt(np.sum((diff * diff.conj()).real)/tot)
