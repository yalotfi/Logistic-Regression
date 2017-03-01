import numpy as np
import process_csv as data
from scipy.optimize import fmin_bfgs
# from scipy.optimize import minimize


def sigmoid(z, derivative=False):
    if not derivative:
        return 1 / (1 + np.exp(-z))
    else:
        return z * (1 - z)


def hypothesis(theta, X_train):
    return sigmoid(X_train.dot(theta.T))


def cost_function(theta, X_train, y_train):
    # Initialize helper variables
    m = len(y_train)
    alpha = 1 / m

    # Cost Function
    pred = hypothesis(theta, X_train)
    neg_case = -y_train.T.dot(np.log(pred))
    pos_case = (1 - y_train.T).dot(np.log(1 - pred))

    # Return cost
    cost = alpha * (neg_case - pos_case)
    return cost[0][0]


def compute_grad(theta, X_train, y_train):
    # Initialize helper variables
    m = len(y_train)
    # alpha = 1 / m
    pred = hypothesis(theta, X_train)
    diff = pred - y_train

    grad = np.zeros(theta.size)
    for i in range(theta.size):
        sum_diff = diff.T.dot(X_train[:, i])
        grad[i] = 1 / m * sum_diff * -1

    return grad


def main():
    file_path = 'grades.txt'
    [X_train, X_test,
     y_train, y_test, theta] = data.process_csv(file_path, 0)

    # initial_theta = np.array(theta)
    # my_args = (X_train, y_train)
    # min_theta = fmin_bfgs(cost_function,
    #     x0=initial_theta,
    #     args=my_args,
    #     maxiter=400,
    #     fprime=compute_grad)

    print('X_train: ', X_train.shape, type(X_train))
    print('y_train: ', y_train.shape, type(y_train))
    print('Theta: ', theta.size, theta.T.shape)
    # print('Optimal Parameters: ', min_theta)
    # print('Min_Cost: ', cost)


if __name__ == '__main__':
    main()
