import numpy as np
from sigmoid import _sigmoid
from data import _process_data


class BasicLogistic():
    '''
    Class to perform basic logistic regression.

    Currently has Methods for:
        - Logistic Sigmoid function
        - Hypothesis Function
        - Cost function
        - Gradient Descent

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

    def cost_function(self, theta):
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
        reg_cost = self.reg_lamda / (2 * self.m) * theta.T.dot(theta)
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

    def optimize(self):
        pass

    def map_feature(self, factor):
        pass

    def plot_boundary(self):
        pass


def main():
    # STEP 1: Process data
    file_path = 'Data/grades.txt'
    [X_train, y_train, init_theta] = _process_data(file_path)

    # STEP 2: Hyperparameters
    # alpha = 0.000001
    reg_lambda = 0  # Lambda value for regularization, if needed

    # Console logs to confirm data structure
    print('X_train: {0} // X_train.T: {1}'.format(
        X_train.shape, X_train.T.shape))
    print('y_train: {0} // y_train.T: {1}'.format(
        y_train.shape, y_train.T.shape))
    print('Theta: {0} // Theta.T: {1}'.format(
        init_theta.shape, init_theta.T.shape))

    # Step 3: Training
    lr = BasicLogistic(init_theta, X_train, y_train, reg_lambda)

    # pred = lr.hypothesis(init_theta)
    # print('Hypothesis Shape: ', pred.shape)

    # cost = lr.cost_function(init_theta)
    # print('Min_Cost: ', cost)

    grad = lr.compute_gradient(init_theta)
    print('Min_Cost: ', grad)


if __name__ == '__main__':
    main()
