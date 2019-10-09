from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    """
    Generate a new feature matrix consisting of all polynomial combinations of 
    the features with degree less than or equal to the specified degree. 
    For example, if an input sample is two dimensional and of the form [a, b], 
    the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
    """
    n_samples, n_features = np.shape(X)
