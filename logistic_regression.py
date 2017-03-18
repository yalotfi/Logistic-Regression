import numpy as np
import sigmoid as sm
from scipy.optimize import fmin


def process_data(file_path):
    '''
    Process the raw data set into 2D arrays of training and label examples.
    This function should be expanded to include training, testing, and
    cross validation sets. It also produces inital theta, outputting three
    arrays.
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

    # Console logs to confirm data structure
    print('\nLog data sizes:')
    print('Examples: {0}\nX transpose: {1}\n'.format(
        X.shape, X.T.shape))
    print('Labels: {0}\ny transpose: {1}\n'.format(
        y.shape, y.T.shape))
    print('Initial thetas: {0}\ntheta Transpose: {1}'.format(
        init_theta.shape, init_theta.T.shape))

    return X, y, init_theta


def hypothesis(theta, X_train):
    '''
    The hypothesis function takes parameters theta and examples X to produce
    a predicted class.
    '''
    return sm.sigmoid(X_train.dot(theta))


def compute_cost(theta, X_train, y_train, reg_lambda=0):
    # Initialize helper variables
    m = len(y_train)  # Number of traning labels
    const = (1 / m)  # Cost Function constant

    # Compute Cost
    pred = hypothesis(theta, X_train)
    loss00 = -y_train.T.dot(np.log(pred))
    loss01 = (1 - y_train.T).dot(np.log(1 - pred))
    reg_cost = (reg_lambda / (2 * m)) * theta.T.dot(theta)
    cost = const * (loss00 - loss01) + reg_cost

    # Return cost
    return float(cost)


def compute_grad(theta, X_train, y_train, reg_lambda=0):
    # Initialize helper variables
    m = len(y_train)  # Number of traning labels
    const = (1 / m)  # Cost Function constant

    # Compute Gradient Step
    pred = hypothesis(theta, X_train)
    loss = pred - y_train
    step = const * X_train.T.dot(loss)

    # Add regularization term
    reg_grad = (reg_lambda / m) * theta[1:, :]
    grad00 = step[0, :]
    grad01 = step[1:, :] + reg_grad
    grads = np.append(grad00, grad01)

    # Sanity Check over vector operations
    print('\nCheck vector operations match up:')
    print('Gradient Size: ', step.shape)  # Should be (n + 1, 1)
    print('Reg Term: ', reg_grad.shape)  # Should be (n, 1)
    print('First Gradient: ', grad00.shape)  # Should be (1,) - bias term
    print('Other Gradients: ', grad01.shape)  # Same as reg_grad
    print('All Gradients: ', grads.shape)  # Same as step

    # Return output
    return grads


def training(theta, X_train, y_train, reg_lambda=0, max_iters=400):
    optimized = fmin(
        compute_cost,
        x0=theta,
        args=(X_train, y_train, reg_lambda),
        maxiter=max_iters,
        full_output=True)
    return optimized[0], optimized[1]


def main():
    # STEP 1: Process data
    file_path = 'Data/grades.txt'
    X, y, init_theta = process_data(file_path)
    # [X_train, X_test,
    #  y_train, y_test] = data.process_csv(file_path, test_size=0)

    # STEP 2: Hyperparameters
    reg_lambda = 0
    # alpha = 0.000001
    max_iters = 400

    cost = compute_cost(init_theta, X, y, reg_lambda=reg_lambda)
    grad = compute_grad(init_theta, X, y, reg_lambda=reg_lambda)

    print('\nLog cost of starting parameters, theta:')
    print('Min_Cost: {0} and Initial gradients: {1}'.format(
        cost, grad)
    )

    print('\nLog data structure of starting cost and theta:')
    print('Data Types for Cost: {0} and Gradient: {1}'.format(
        type(cost), type(grad))
    )

    # Step 3: Training
    print('\nTraining Model:')
    theta, min_cost = training(init_theta, X, y, reg_lambda, max_iters)
    print('Optimized Thetas: {0}\nMinimum Cost: {1}'.format(theta, min_cost))


if __name__ == '__main__':
    main()
