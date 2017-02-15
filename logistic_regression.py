import numpy as np
from sklearn.cross_validation import train_test_split


class BasicLogistic():
    '''
    Class to perform basic logistic regression.

    Currently has Methods for:
        - Logistic Sigmoid function
        - Hypothesis Function
        - Cost function
        - Compute Gradient

    Need Methods for:
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
        Logistic Sigmoid function used for building classifiers. The
        Derivative of the sigmoid is necessary in Neural Network classifiers.
        '''
        # Compute derivative or not
        if not derivative:
            return 1 / (1 + np.exp(-z))
        else:
            return z * (1 - z)
    def hypothesis(self, X_train, theta):
        return self.sigmoid(X_train * theta.T)

    def regularization(self, theta, reg_lamda):
        pass

    def cost_function(self, X_train, y_train, theta):
        '''
        Cost function J(theta) to be minimized. To perform linear algebra,
        it's best if the data is of type matrix. If data is passed as a numpy
        array, the function will just convert it to a matrix. You can still
        use dot products to perform the same operations on arrays, but this
        conversion produces cleaner code. Writing vectorized code is less
        error prone, easy to read, and usually faster than iterative loops.
        '''
        # Initialize helper variables
        m = len(y_train)  # Number of traning examples
        alpha = (1 / m)  # Cost Function constant

        # Vectorized Cost Function              # Intuition:
        pred = self.hypothesis(X_train, theta)  # (70x3)*(3x1)=(70x1)
        neg_case = -y_train.T * np.log(pred)  # (1x70)*(70*1)=(1x1)
        pos_case = (1 - y_train.T) * np.log(1 - pred)  # (1x70)*(70x1)=(1x1)

        # Return Cost
        cost = alpha * (neg_case - pos_case)  # (1x1)-(1x1)=(1x1)
        return cost

    def compute_grad(self, X_train, y_train, theta):
        '''
        Compute the gradient, or partial derivative, of the calculated cost.
        This is used to optimize the learning algorithm's parameters that fit
        some prediction or hypothesis function to the data. Minimizing the
        cost by an optimization function is basically searching for the global
        minimum of the function. How best to do that is up for debate.
        '''
        m = len(X_train)
        pred = self.hypothesis(X_train, theta)
        diff = pred - y_train
        grad = (1/m) * X_train.T * diff
        
        return grad

    def map_feature(X):
        pass


def process_csv(file_path, test_size=0.3):
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

    # Split training set and coerce to matrices
    [X_train, X_test,
     y_train, y_test] = train_test_split(X, y, test_size=test_size)

    # Add intercept column to training examples
    m, n = X_train.shape  # Dimensions, m x n of X train examples
    ones = np.ones(m, dtype='int')  # Vector of ones
    col_vecs = []  # init a list which will store x columns
    for col in range(n):  # n is number of columns across X_train
        col_vecs.append(X_train[:, col])  # List cols of X_train for stacking
    X_train = np.vstack((ones, col_vecs)).T  # Create tuple and bind X_train

    # Initialize parameters, theta
    theta = np.matrix([[0 for param in range(n + 1)]])

    # Convert to numpy matrices:
    X_train = np.matrix(X_train)
    y_train = np.matrix(y_train)
    X_test = np.matrix(X_test)
    y_test = np.matrix(y_test)

    # Return processed data
    return [X_train, X_test,
            y_train, y_test, theta]  # Return train, test, and parameter sets


def main():
    # STEP 1: Process data
    file_path = 'grades.txt'
    [X_train, X_test,
     y_train, y_test, theta] = process_csv(file_path)

    # STEP 2: Hyperparameters
    # theta = np.matrix([[0 for param in range(n + 1)]])  # Initialize params
    reg_lambda = 1  # Lambda value for regularization, if needed

    # Console Logs for Testing
    print('X Training Set: ', X_train.shape)
    print('y training Set: ', y_train.shape)
    print('y transpose: ', np.transpose(y_train).shape)
    print('Params: {0} Size: {1}'.format(theta, theta.shape))
    print('Lamda set to: ', reg_lambda)
    print(type(X_train))
    print(X_train[0:10, :])

    # Step 3: Training
    lr = BasicLogistic(X_train, y_train, theta, reg_lambda)
    print(lr.sigmoid((X_train * theta)[0:10, :]))


if __name__ == '__main__':
    main()
