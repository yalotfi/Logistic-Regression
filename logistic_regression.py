import numpy as np


class BasicLogistic():
    """
    Object to perform basic logistic regression.

    Currently has Methods for:
        - Logistic Sigmoid function
        - Hypothesis function
        - Cost function

    Need Methods for:
        - Regularization
        - Gradient Descent
    """
    def __init__(self, z, X_train, y_train):
        super().__init__()
        self.z = z
        self.X_train
        self.y_train

    def sigmoid(self, z, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-self.z))
        else:
            return self.z * (1 - self.z)

    def hypothesis(self, X_train, theta):
        '''
        Hypothesis function (prediction method)
        '''

        # Simple predictor
        return self.sigmoid(self.X_train * self.theta)

    def cost_function(self, theta, X_train, y_train):
        '''
        Cost function J(theta) to be minimized
        '''

        # Initialize variables
        m = len(self.y_train)  # Number of traning examples
        preds = self.hypothesis(X_train, theta)

        # Vectorized cost function (no iteration needed)
        cost = -np.transpose(self.y_train) * np.log(preds)
        cost -= (1 - np.transpose(y_train)) * np.log(1 - (preds))
        cost *= (1 / m)

        # Return unregularized costs
        return cost


def main():
    # Disclaimer!
    print('Still Hacking...')

    # Temp data to pass through model
    z_test = np.array([[1, 2], [3, 4]])
    print(z_test)
    print(type(z_test))

    # Testing Method
    lr = BasicLogistic(z_test)
    print(lr.sigmoid(z_test))


if __name__ == '__main__':
    main()
