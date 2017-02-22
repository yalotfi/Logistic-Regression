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
        return self.sigmoid(self.X_train.dot(self.theta.T))

    def cost_function(self):
        '''
        Cost function J(theta) to be minimized. To perform linear algebra,
        it's best if the data is of type matrix. If data is passed as a numpy
        array, the function will just convert it to a matrix. You can still
        use dot products to perform the same operations on arrays, but this
        conversion produces cleaner code. Writing vectorized code is less
        error prone, easy to read, and usually faster than iterative loops.
        '''
        # Check data types, convert to matrices
        # X_train = np.array(X_train)
        # y_train = np.matrix(y_train)
        # theta = np.matrix(theta)

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

    def compute_grad(self):
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
