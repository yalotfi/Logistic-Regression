import numpy as np
from sklearn.cross_validation import train_test_split


class BasicLogistic():
    '''
    Object to perform basic logistic regression.

    Currently has Methods for:
        - Logistic Sigmoid function
        - Cost function

    Need Methods for:
        - Regularization
        - Gradient Descent
    '''
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

    def cost_function(self, X_train, y_train, theta):
        ''' Cost function J(theta) to be minimized '''
        # Initialize variables
        m = len(y_train)  # Number of traning examples

        # Compute cost function
        for i in range(0, m):
            pred = self.sigmoid(theta[i, 0] * X_train[i])  # Hypothesis
            cost = -y_train[i] * np.math.log(pred[i])
            cost -= (1 - y_train[i]) * np.math.log(1 - pred[i])
            cost *= (1/m)

        # Return unregularized costs
        return cost

    def map_feature(X, order):
        pass

    def regularization(self, reg_lamda):
        pass


def process_csv(file_path):
    # Load Grade Data
    raw_data = np.genfromtxt(file_path, delimiter=',')
    X = raw_data[:, 0:2]
    y = raw_data[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return [X_train, X_test, y_train, y_test]


def main():
    # Load processed data
    file_path = 'grades.txt'
    [X_train, X_test, y_train, y_test] = process_csv(file_path)

    # Hyperparameters
    m = len(X_train)  # length of the training set
    theta = np.array([0 for rows in range(m)])  # Initialize parameters at 0
    reg_lambda = 1  # Lambda value for regularization, if needed

    # Console Logs for Testing
    print('X Training Set: ', X_train.shape)
    print('y training Set: ', y_train.shape)
    print('Initial Parameters: ', theta.shape)
    print('Lamda set to: ', reg_lambda)
    print(type(X_train))

    # Testing Class Methods
    lr = BasicLogistic(X_train, y_train, theta, reg_lambda)
    print(lr.sigmoid(X_train))
    print(lr.cost_function(X_train, y_train, theta))


if __name__ == '__main__':
    main()
