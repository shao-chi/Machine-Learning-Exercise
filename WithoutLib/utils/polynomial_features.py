from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    """
    Generate a new feature matrix consisting of all polynomial combinations of 
    the features with degree less than or equal to the specified degree. 
    For example, if an input sample is two dimensional and of the form [a, b], 
    the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
    """
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new
