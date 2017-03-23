import numpy as np
from scipy.optimize import fmin
from sigmoid import _sigmoid
from data import _process_data


class BasicLogistic():
    '''
    Class to perform basic logistic regression.

    Currently has Methods for:
        - Logistic Sigmoid function
        - Hypothesis Function
        - Cost function
        - Optimization

    Need Methods for:
        - Regularization
        - Feature Mapping
    '''

    def __init__(self, init_theta, X_train, y_train, reg_lambda):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.init_theta = init_theta
        self.reg_lambda = reg_lambda
        self.m = len(self.y_train)

    def hypothesis(self, theta):
        '''
        The hypothesis function predicts the value of y given input x and
        parameters theta. In classification, predictions are passed through
        a sigmoidal function where values closer to 1 have a higher
        probability of being 1 and vice versa for values closer to zero.
        This function is used in calculating the error and partial derivative
        for fitting the best decision boundary on the given data set.
        '''
        return _sigmoid(self.X_train.dot(theta))

    def compute_cost(self, theta):
        '''
        Cost function J(theta) to be minimized. Calculating the total error
        is done by finding the distance from the actual label. In this case,
        if the label is 1, then the negative case is computed. Conversely, if
        the label is 0, then the positive case is computed. The magic of this
        cost function is that it represents either case in one equation.
        '''
        const = (1 / self.m)  # Cost Function constant

        # Cost Function
        pred = self.hypothesis(theta)
        loss00 = -self.y_train.T.dot(np.log(pred))
        loss01 = (1 - self.y_train.T).dot(np.log(1 - pred))
        reg_cost = self.reg_lambda / (2 * self.m) * theta.T.dot(theta)
        cost = const * (loss00 - loss01) + reg_cost

        # Return Cost
        return float(cost)

    def compute_gradient(self, theta):
        const = (1 / self.m)

        # Gradient Step
        pred = self.hypothesis(theta)
        loss = pred - self.y_train
        step = const * self.X_train.T.dot(loss)

        # Add regularization term
        reg_grad = self.reg_lambda / self.m * theta[1:, :]
        grad00 = step[0, :]
        grad01 = step[1:, :] + reg_grad
        grads = np.append(grad00, grad01)

        return grads

    def optimize(self, theta, max_iters=400):
        optimized = fmin(
            self.compute_cost,
            x0=theta,
            maxiter=400,
            full_output=True)
        return optimized[0], optimized[1]

    def map_feature(self, factor):
        pass

    def plot_boundary(self):
        pass


def main():
    # STEP 1: Process data
    file_path = 'Data/grades.txt'
    X_train, y_train, init_theta = _process_data(file_path)

    # STEP 2: Hyperparameters
    reg_lambda = 0  # Regularization term, lambda
    max_iters = 400  # For optimization

    lr = BasicLogistic(init_theta, X_train, y_train, reg_lambda)
    cost = lr.compute_cost(init_theta)
    grad = lr.compute_gradient(init_theta)

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
    theta, min_cost = lr.optimize(init_theta, max_iters)
    print('Optimized Thetas: {0}\nMinimum Cost: {1}'.format(theta, min_cost))


if __name__ == '__main__':
    main()
