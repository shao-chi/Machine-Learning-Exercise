def normalization(X, axis=-1, order=2):
    """
    Scaling to unit length
    x = x / ||x||
    ||x|| = 1
    """
    norm = np.atleast_1d(np.linalg.norm(X, order, axis))
    norm[norm == 0] = 1
    return X / np.expand_dims(norm, axis)