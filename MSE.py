def mse(test, predict):
    """
    evaluate the mean squared error between true data and predicted data
    """
    result = 0
    for t, p in zip(test, predict):
        result += (p - t) ** 2
    result /= len(test)
    return result