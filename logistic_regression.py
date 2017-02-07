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
        # self.z = z
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

    # Load Grade Data
    grade_data = np.genfromtxt('grades.txt', delimiter=',')
    m = len(grade_data)

    # Load Training Data
    X_train = grade_data[:, 0], grade_data[:, 1]
    y_train = grade_data[:, 2]

    # Initialize parameters at 0 (okay for linear/logistic regression)
    theta = np.array([0 for rows in range(m)])

    # Lambda value for regularization, if needed
    reg_lambda = 0.1

    # Console Logs for Testing
    print('Raw Data Size: ', grade_data.shape)
    print('X Training Set: ', X_train.shape)
    print('y training Set: ', y_train.shape)
    print('Initial Parameters: ', theta.shape)
    print('Lamda set to: ', reg_lambda)
    print(type(X_train))

    # Testing Class Methods
    '''
    lr = BasicLogistic(X_train, y_train, theta, reg_lambda)
    print(lr.sigmoid(X_train))
    print(lr.hypothesis(X_train, theta))
    print(lr.cost_function(X_train, y_train, theta))
    '''


if __name__ == '__main__':
    main()
