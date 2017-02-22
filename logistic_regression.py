import numpy as np
import process_csv
from scipy.optimize import fmin_bfgs


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
    def hypothesis(self):
        '''
        The hypothesis function predicts the value of y given input x and
        parameters theta. In classification, predictions are passed through
        a sigmoidal function where values closer to 1 have a higher
        probability of being 1 and vice versa for values closer to zero.
        This function is used in calculating the error and partial derivative
        for fitting the best decision boundary on the given data set.
        '''
        return self.sigmoid(self.X_train.dot(self.theta.T))

    def cost_function(self, theta, X_train, y_train):
        '''
        Cost function J(theta) to be minimized. Calculating the total error
        is done by finding the distance from the actual label. In this case,
        if the label is 1, then the negative case is computed. Conversely, if
        the label is 0, then the positive case is computed. The magic of this
        cost function is that it represents either case in one equation.
        '''
        # Initialize helper variables
        m = len(self.y_train)  # Number of traning labels
        alpha = (1 / m)  # Cost Function constant

        # Vectorized Cost Function              # Intuition:
        pred = self.hypothesis()  # #                (70x3)*(3x1)=(70x1)
        neg_case = -self.y_train.T.dot(np.log(pred))  # (1x70)*(70*1)=(1x1)
        pos_case = (1 - self.y_train.T).dot(np.log(1 - pred))  # (1x70)*(70x1)=(1x1)

        # Return Cost
        cost = alpha * (neg_case - pos_case)  # (1x1)-(1x1)=(1x1)
        return cost[0][0]  # Return single value instead of array

    def compute_grad(self, theta, X_train, y_train):
        '''
        Compute the gradient, or partial derivative, of the calculated cost.
        This is used to optimize the learning algorithm's parameters that fit
        some prediction or hypothesis function to the data. Minimizing the
        cost by an optimization function is basically searching for the global
        minimum of the function.
        '''
        m = len(self.X_train)
        pred = self.hypothesis()
        diff = pred - self.y_train

        grad = np.zeros(self.theta.size)
        for i in range(self.theta.size):
            sum_diff = diff.T.dot(self.X_train[:, i])
            grad[i] = (1/m) * sum_diff
        
        return grad

    def optimize(self):
        my_args = (self.X_train, self.y_train)
        return fmin_bfgs(self.cost_function, 
                         x0=self.theta, 
                         args=my_args, 
                         maxiter=400, 
                         fprime=self.compute_grad)

    def map_feature(X):
        pass

    def regularization(self):
        pass


def main():
    # STEP 1: Process data
    file_path = 'grades.txt'
    [X_train, X_test,
     y_train, y_test, theta] = process_csv(file_path)

    # STEP 2: Hyperparameters
    # theta = np.matrix([[0 for param in range(n + 1)]])  # Initialize params
    reg_lambda = 0  # Lambda value for regularization, if needed

    # Console Logs for Testing
    print('X Training Set: ', X_train.shape)
    print('y training Set: ', y_train.shape)
    print('y transpose: ', np.transpose(y_train).shape)
    print('Params: {0} Size: {1}'.format(theta, theta.shape))
    print('Lamda set to: ', reg_lambda)

    # Step 3: Training
    lr = BasicLogistic(X_train, y_train, theta, reg_lambda)

    # Optimization
    initial_theta = np.array(theta)
    opt_theta = fmin_bfgs(lr.cost_function(X_train, y_train, theta), 
        initial_theta,
        fprime=lr.compute_grad(X_train, y_train, theta),
        maxiter=400
    )

    print(opt_theta)



if __name__ == '__main__':
    main()
