import numpy as np
from sklearn.cross_validation import train_test_split


def _process_data(file_path):
    '''
    Process the raw data set into 2D arrays of training and label examples.
    This function should be expanded to include training, testing, and
    cross validation sets. It also produces inital theta, outputting three
    arrays.

    **Important: This function will append an intercept column of ones.**
    '''
    # Read in the Data
    raw_data = np.genfromtxt(file_path, delimiter=',')

    # Examples happen to be in first two cols and labels in the third
    X = np.array(raw_data[:, :-1])
    y = np.array(raw_data[:, -1:])

    # Save dimensions
    m, n = X.shape

    # Insert intercept column for linear algebra
    X = np.insert(X, 0, 1, axis=1)

    # Initial parameters set to zero defined by shape (3x1)
    init_theta = np.zeros(((n + 1), 1))

    return X, y, init_theta


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

    # Return processed data
    return [X_train, X_test,
            y_train, y_test]  # Return train, test, and parameter sets
