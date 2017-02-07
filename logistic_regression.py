import numpy as np


class BasicLogistic():
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
        
    def sigmoid(self, z, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-z))
        else:
            return z * (1 - z)


def main():
    # Disclaimer!
    print('Still Hacking...')

    # Temp data to pass through model
    z_test = np.array([[1,2],[3,4]])
    print(z_test)
    print(type(z_test))

    # Testing Method
    lr = BasicLogistic(z_test)
    print(lr.sigmoid(z_test))


if __name__ == '__main__':
    main()