import numpy as np
import logistic_regression as lr

file_path = 'grades.txt'
[X_train, X_test,
 y_train, y_test, m, n] = lr.process_csv(file_path, 0)

theta = np.matrix([[0 for param in range(n + 1)]])
reg_lambda = 1

lrCost = lr.BasicLogistic(X_train, y_train, theta, reg_lambda)
cost = lrCost.cost_function(X_train, y_train, theta)

#print(X_train)
print("X_train: ", X_train.shape, type(X_train))
print("y_train: ", y_train.shape, type(y_train))
print('Theta: ', theta.shape, type(theta))
print('Matrix, X, dimensions: ', m, n)
#print(X_train * theta.T)
#print(lrCost.sigmoid(X_train * theta.T))
print("Initial Cost to Minimize: ", cost)