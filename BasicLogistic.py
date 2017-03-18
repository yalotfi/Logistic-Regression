import numpy as np
import sigmoid as sm
import process_csv as data


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

    def __init__(self, init_theta, X_train, y_train, reg_lamda):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.init_theta = init_theta
        self.reg_lamda = reg_lamda

    def hypothesis(self, theta):
        '''
        The hypothesis function predicts the value of y given input x and
        parameters theta. In classification, predictions are passed through
        a sigmoidal function where values closer to 1 have a higher
        probability of being 1 and vice versa for values closer to zero.
        This function is used in calculating the error and partial derivative
        for fitting the best decision boundary on the given data set.
        '''
        return sm.sigmoid(self.X_train.dot(theta.T))

    def cost_function(self, theta):
        '''
        Cost function J(theta) to be minimized. Calculating the total error
        is done by finding the distance from the actual label. In this case,
        if the label is 1, then the negative case is computed. Conversely, if
        the label is 0, then the positive case is computed. The magic of this
        cost function is that it represents either case in one equation.
        '''
        # Initialize helper variables
        m = len(self.y_train)  # Number of traning labels
        const = (1 / m)  # Cost Function constant

        # Cost Function
        pred = self.hypothesis(theta)
        neg_case = -self.y_train.T.dot(np.log(pred))
        pos_case = (1 - self.y_train.T).dot(np.log(1 - pred))

        # Return Cost
        cost = const * (neg_case - pos_case)  # (1x1)-(1x1)=(1x1)

        # Compute gradient Step
        pred = self.hypothesis(theta)
        grad = 1 / m * self.X_train.T.dot(pred - self.y_train).T

        return [cost[0][0], grad[0]]

    def gradient_descent(self, alpha, max_iters):
        '''
        Compute the gradient, or partial derivative, of the calculated cost.
        This is used to optimize the learning algorithm's parameters that fit
        some prediction or hypothesis function to the data. Minimizing the
        cost by an optimization function is basically searching for the global
        minimum of the function.
        '''
        m = float(len(self.X_train))
        J_history = [0] * max_iters
        theta = self.init_theta

        for i in range(max_iters):
            pred = self.hypothesis(theta)
            gradient = self.X_train.T.dot(pred - self.y_train)
            theta = theta - alpha / m * gradient.T

            cost = self.cost_function(theta)
            J_history[i] = cost

        return theta, np.array(J_history)

    def map_feature(X):
        pass

    def regularization(self):
        pass


def main():
    # STEP 1: Process data
    file_path = 'Data/grades.txt'
    [X_train, X_test,
     y_train, y_test, theta] = data.process_csv(file_path)

    # STEP 2: Hyperparameters
    alpha = 0.000001
    reg_lambda = 0  # Lambda value for regularization, if needed

    # Console logs to confirm data structure
    print('X_train: {0} // X_train.T: {1}'.format(
        X_train.shape, X_train.T.shape))
    print('y_train: {0} // y_train.T: {1}'.format(
        y_train.shape, y_train.T.shape))
    print('Theta: {0} // Theta.T: {1}'.format(
        theta.shape, theta.T.shape))

    # Step 3: Training
    lr = BasicLogistic(theta, X_train, y_train, reg_lambda)

    # pred = lr.hypothesis(theta)
    # print('Hypothesis Shape: ', pred.shape)

    # cost, grad = lr.cost_function(theta)
    # print('Min_Cost: ', cost, grad)


if __name__ == '__main__':
    main()
