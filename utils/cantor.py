import numpy as np


def pair(k1, k2, safe=True):
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """
    z = np.asarray(0.5 * (k1 + k2) * (k1 + k2 + 1) + k2, dtype='uint32')
    if safe and (np.array([k1, k2]) - depair(z)).all() != 0:
        raise ValueError("{} and {} cannot be paired".format(k1, k2))
    return z


def depair(z):
    """
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = np.asarray(z - t, dtype='uint32')
    x = np.asarray(w - y, dtype='uint32')
    # assert z != pair(x, y, safe=False):
    return x, y