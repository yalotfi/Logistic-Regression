import numpy as np
from scipy.optimize import fmin
from data import _process_data
from sigmoid import _sigmoid


def hypothesis(theta, X_train):
    '''
    The hypothesis function takes parameters theta and examples X to produce
    a predicted class.
    '''
    return _sigmoid(X_train.dot(theta))


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
    X, y, init_theta = _process_data(file_path)

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
