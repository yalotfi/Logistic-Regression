import numpy as np


class BasicLogistic():
    """
    Object to perform basic logistic regression.

    Currently has Methods for:
        - Sigmoid function

    Need Methods for:
        - Cost function
        - Regularization
        - Gradient Descent
        - Prediction function
    """
    def __init__(self, z, X_train, y_train, learn_rate):
        super().__init__()
        self.z = z
        self.X_train
        self.y_train
        self.learn_rate
        
    def sigmoid(self, z, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-self.z))
        else:
            return self.z * (1 - self.z)

    def hypothesis(self, X_train, theta):
        ''' Hypothesis function (prediction method) '''

    def cost_function(self, X_train, y_train, learn_rate):
        # Number of traning examples
        m = len(self.y_train)

        preds = self.sigmoid(self.X * self.theta)


def main():
    # Disclaimer!
    print('Still Hacking...')

    # Temp data to pass through model
    z_test = np.array([[1,2],[3,4]])
    print(z_test)
    print(type(z_test))

    # Testing Method
    lr = BasicLogistic(z_test)
    print(lr.sigmoid(z_test))


if __name__ == '__main__':
    main()