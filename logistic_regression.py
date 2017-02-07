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
    def __init__(self, X_train, y_train, theta, reg_lamda):
        super().__init__()
        #self.z = z
        self.X_train = X_train
        self.y_train = y_train
        self.theta = theta
        self.reg_lamda = reg_lamda

    def sigmoid(self, z, derivative=False):
        ''' Logistic Sigmoid function classifies binary values '''
        # Compute derivative or not
        if not derivative:
            return 1 / (1 + np.exp(-z))
        else:
            return z * (1 - z)

    def hypothesis(self, X_train, theta):
        ''' Hypothesis function (prediction method) '''
        # Simple predictor function
        return self.sigmoid(X_train * theta)

    def cost_function(self, X_train, y_train, theta):
        ''' Cost function J(theta) to be minimized '''
        # Initialize variables
        m = len(self.y_train)  # Number of traning examples
        preds = self.hypothesis(X_train, theta)

        # Vectorized cost function (no iteration needed)
        cost = -np.transpose(y_train) * np.log(preds)
        cost -= (1 - np.transpose(y_train)) * np.log(1 - (preds))
        cost *= (1 / m)

        # Return unregularized costs
        return cost

    def regularization(self, reg_lamda):
        pass


def main():
    # Disclaimer!
    print('Still Hacking...')

    # Temp data to pass through model
    X_train = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
    y_train = np.transpose(np.array([1, 0, 1]))
    theta = np.transpose(np.array([0, 0, 0]))
    reg_lambda = 0.1
    print('X Training Set: ', X_train)
    print('y training Set: ', y_train)
    print('Initial Parameters: ', theta)
    print('Lamda set to: ', reg_lambda)
    print(type(X_train))
    print(X_train.shape)

    # Testing Method
    lr = BasicLogistic(X_train, y_train, theta, reg_lambda)  # params: X_train, y_train, theta, reg_lambda
    print(lr.sigmoid(X_train))
    print(lr.hypothesis(X_train, theta))
    print(lr.cost_function(X_train, y_train, theta))


if __name__ == '__main__':
    main()
