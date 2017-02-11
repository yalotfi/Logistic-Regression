import numpy as np
from sklearn.cross_validation import train_test_split


class BasicLogistic():
    '''
    Object to perform basic logistic regression.

    Currently has Methods for:
        - Logistic Sigmoid function
        - Cost function

    Need Methods for:
        - Gradient Descent
        - Regularization
        - Feature Mapping
    '''
    def __init__(self, X_train, y_train, theta, reg_lamda):
        super().__init__()
        # self.z = z
        self.X_train = X_train
        self.y_train = y_train
        self.theta = theta
        self.reg_lamda = reg_lamda

    def sigmoid(self, z, derivative=False):
        ''' 
        Logistic Sigmoid function used for logistic regression and 
        Neural Network Classifiers.  Derivative of the sigmoid used
        for the latter.
        '''
        # Compute derivative or not
        if not derivative:
            return 1 / (1 + np.exp(-z))
        else:
            return z * (1 - z)

    def cost_function(self, X_train, y_train, theta):
        ''' Cost function J(theta) to be minimized '''
        # Initialize variables
        m = len(y_train)  # Number of traning examples

        # Vectorized Cost Function
        alpha = (1 / m)
        pred = self.sigmoid(X_train * theta)
        neg_case = -y_train.T * np.log(pred)
        pos_case = (1 - y_train.T) * np.log(1 - pred)

        # Return Cost
        cost = alpha * (neg_case - pos_case)
        return cost

    def min_cost(self):
        pass

    def map_feature(X, order):
        pass

    def regularization(self, reg_lamda):
        pass


def process_csv(file_path, test_size=0.5):
    '''
    1) Load data from a file path and split into train and test sets based on
    specified test ratio which is 50-50 by default.

    2) Adds an intercept column to the X training examples to make the math
    cleaner... Python does not make matrix operations easy.
    '''
    # Load Grade Data
    raw_data = np.genfromtxt(file_path, delimiter=',')
    m_raw, n_raw = raw_data.shape  # Dims of raw_data

    # Seperate labels and examples
    X = np.zeros((m_raw, n_raw - 1))  # Exclude label column
    y = np.zeros((m_raw, 1))  # init label vector
    for col in range(n_raw):  # Over all columns in array
        bin_flag = np.unique(raw_data[:, col]) == [0, 1]
        if type(bin_flag) is bool:  # flag is False if binary, list if True
            X[:, col] = raw_data[:, col]  # Assign to X examples
        else:  # If flag matches binary test, it return a list of Trues
            y[:, 0] = raw_data[:, col]  # Assign to y labels

    # Split
    [X_train, X_test,
     y_train, y_test] = train_test_split(X, y, test_size=test_size)

    # Add intercept column to training examples
    m, n = X_train.shape  # Dimensions, m x n of X train examples
    ones = np.ones(m, dtype='int')  # Vector of ones
    col_vecs = []  # init a list which will store x columns
    for col in range(n):  # n is number of columns across X_train
        col_vecs.append(X_train[:, col])  # List cols of X_train for stacking
    X_train = np.vstack((ones, col_vecs)).T  # Create tuple and bind X_train

    # Return processed data
    return [X_train, X_test,
            y_train, y_test, m, n]  # Return Train and Test sets & X train dim


def main():
    # STEP 1: Process data
    file_path = 'grades.txt'
    [X_train, X_test,
     y_train, y_test, m, n] = process_csv(file_path)

    # STEP 2: Hyperparameters
    theta = np.array([[0 for param in range(n + 1)]])  # Initialize parameters
    reg_lambda = 1  # Lambda value for regularization, if needed

    # Step 3: Training
    '''
    lr = BasicLogistic(X_train, y_train, theta, reg_lambda)
    print(lr.sigmoid(X_train))
    print(lr.cost_function(X_train, y_train, theta))
    '''

    # Console Logs for Testing
    print('X Training Set: ', X_train.shape)
    print('y training Set: ', y_train.shape)
    print('y transpose: ', np.transpose(y_train).shape)
    print('Params: {0} Size: {1}'.format(theta, theta.shape))
    print('Lamda set to: ', reg_lambda)
    print(type(X_train))


if __name__ == '__main__':
    main()
