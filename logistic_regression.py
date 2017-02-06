import numpy as np


class BasicLogistic(object):
    """
    Object to perform basic logistic regression.

    Currently has Methods for:
        - Sigmoid function

    Need Methods for:
        - Cost function
        - Regularization
        - Gradient Descent
        - Prediction function
    """
    def __init__(self, z):
        super().__init__()
        self.z = z
        
    def sigmoid(self, z):
        g = np.array([[0 for cols in range(2)] for rows in range(4)])
        for row in z:
            g = row
        return g


def main():
    print('Still Hacking...')


if __name__ == '__main__':
    main()