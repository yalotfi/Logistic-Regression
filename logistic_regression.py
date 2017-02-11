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


def process_csv(file_path, ratio=0.5):
    '''
    1) Load data from a file path and split into train and test sets based on a 
    specified ratio which set by default to a 50-50 split.

    2) Adds an intercept column to the X training examples to make the math
    cleaner... Python does not make matrix operations easy.
    '''
    # Load Grade Data
    raw_data = np.genfromtxt(file_path, delimiter=',')
    m_raw, n_raw = raw_data.shape  # Dims of raw_data

    # Seperate labels and examples
    X = np.zeros((m_raw, n_raw - 1))  # Exclude label column
    for col in range(n_raw):  # Over all columns in array
        bin_flag = np.unique(raw_data[:, col]) == [0,1]
        if type(bin_flag) is bool:  # flag will be False if binary, list if True
            X[:, col] = raw_data[:, col]  # Assign to X examples
        else: # If flag matches binary test, it return a list of Trues
            y = raw_data[:, col]  # Assign to y labels

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)

    # Add intercept column to training examples
    m, n = X_train.shape  # Dimensions, m x n
    ones = np.ones(m, dtype='int')
    col_vecs = [X_train[:, 0], X_train[:, 1]]
    X_train = np.vstack((ones, col_vecs)).T

    # Return processed data
    return [X_train, X_test, y_train, y_test]


def main():
    # STEP 1: Process data
    file_path = 'grades.txt'
    [X_train, X_test, y_train, y_test] = process_csv(file_path)

    # STEP 2: Hyperparameters
    m, n = X_train.shape
    theta = np.array([0 for rows in range(n + 1)])  # Initialize parameters
    reg_lambda = 1  # Lambda value for regularization, if needed

    # Training
    lr = BasicLogistic(X_train, y_train, theta, reg_lambda)
    print(lr.sigmoid(X_train))
    print(lr.cost_function(X_train, y_train, theta))

    # Console Logs for Testing
    print('X Training Set: ', X_train.shape)
    print('y training Set: ', y_train.shape)
    print('Initial Parameters: ', theta.shape)
    print('Lamda set to: ', reg_lambda)
    print(type(X_train))


if __name__ == '__main__':
    main()
