import numpy as np


def sigmoid(z, derivative=False):
    '''
    Logistic Sigmoid function used for building classifiers. The
    Derivative of the sigmoid is necessary in Neural Network classifiers.
    '''
    # Compute derivative or not
    if not derivative:
        return 1 / (1 + np.exp(-z))
    else:
        return z * (1 - z)
