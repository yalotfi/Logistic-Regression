import numpy as np
import process_csv as data
from scipy.optimize import fmin_bfgs


def normalize(X):
    for col in range(X):
        X[col]
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypothesis(theta, X_train):
    return sigmoid(X_train.dot(theta.T))


def compute_cost(theta, X_train, y_train, reg_lambda=0):
    # Initialize helper variables
    m = len(y_train)  # Number of traning labels
    const = (1 / m)  # Cost Function constant

    # Compute Cost
    pred = hypothesis(theta, X_train)
    loss = -y_train.T.dot(np.log(pred)) - (1 - y_train.T).dot(np.log(1 - pred))
    reg_cost = (reg_lambda / (2 * m)) * theta.dot(theta.T)
    cost = const * loss + reg_cost

    # Return cost
    return cost.flatten()


def compute_grad(theta, X_train, y_train, reg_lambda=0):
    # Initialize helper variables
    m = len(y_train)  # Number of traning labels
    const = (1 / m)  # Cost Function constant

    # Compute Gradient Step
    pred = hypothesis(theta, X_train)
    step = const * X_train.T.dot(pred - y_train).T
    # reg_grad = (reg_lambda / m) * theta[:, 1:]
    # grad01 = step[:, 0]
    # grad02 = step[:, 1:] + reg_grad
    # grad = np.append(grad01, grad02)

    # Return output
    # return grad.flatten()
    return step.flatten()


def training(theta, X_train, y_train, max_iters):
    def f(theta):
        return compute_cost(theta, X_train, y_train)

    def fprime(theta):
        return compute_grad(theta, X_train, y_train)

    opt_theta = fmin_bfgs(f, x0=theta, fprime=fprime, maxiter=max_iters)
    return opt_theta


def main():
    # STEP 1: Process data
    file_path = 'Data/grades.txt'
    [X_train, X_test,
     y_train, y_test] = data.process_csv(file_path, test_size=0)

    # Initialize parameters, theta, and helper vars
    init_theta = np.zeros([1, X_train.shape[1]])

    # STEP 2: Hyperparameters
    reg_lambda = 0
    alpha = 0.000001
    max_iters = 400

    # Console logs to confirm data structure
    print('X_train: {0} // X_train.T: {1}'.format(
        X_train.shape, X_train.T.shape))
    print('y_train: {0} // y_train.T: {1}'.format(
        y_train.shape, y_train.T.shape))
    print('Theta: {0} // Theta.T: {1}'.format(
        init_theta.shape, init_theta.T.shape))
    # print(init_theta[1:])

    cost = compute_cost(init_theta, X_train, y_train, reg_lambda=reg_lambda)
    grad = compute_grad(init_theta, X_train, y_train, reg_lambda=reg_lambda)
    print('Min_Cost: {0} and Initial gradients: {1}'.format(cost, grad))

    # Step 3: Training
    opt_theta_2 = training(init_theta, X_train, y_train, max_iters)
    print('Optimized Thetas: ', opt_theta_2)


if __name__ == '__main__':
    main()
