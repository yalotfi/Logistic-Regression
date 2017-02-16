import logistic_regression as lr
import process_csv


file_path = 'grades.txt'
[X_train, X_test,
y_train, y_test, theta] = process_csv(file_path, 0)

# theta = np.matrix([[0 for param in range(n + 1)]])
reg_lambda = 0

lrCost = lr.BasicLogistic(X_train, y_train, theta, reg_lambda)
cost = lrCost.cost_function(X_train, y_train, theta)
grad = lrCost.compute_grad(X_train, y_train, theta)
opt_theta = lrCost.optimize(X_train, y_train, theta)

#print(X_train)
print('X_train: ', X_train.shape, type(X_train))
print('y_train: ', y_train.shape, type(y_train))
print('Theta: ', theta.shape, type(theta))
print('Initial Cost to Minimize: ', cost)
print('Approx Gradient: ', grad)
print('Optimal Parameters: ', opt_theta)