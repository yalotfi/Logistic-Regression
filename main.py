import logistic_regression as lr
import process_csv as data


file_path = 'grades.txt'
[X_train, X_test,
y_train, y_test, theta] = data.process_csv(file_path, 0)

# theta = np.matrix([[0 for param in range(n + 1)]])
reg_lambda = 0

lrCost = lr.BasicLogistic(X_train, y_train, theta, reg_lambda)
pred = lrCost.hypothesis()
cost = lrCost.cost_function()
grad = lrCost.compute_grad()
# opt_theta = lrCost.optimize(X_train, y_train, theta)

#print(X_train)
print('X_train: ', X_train.shape, type(X_train))
print('y_train: ', y_train.shape, type(y_train))
print('Theta: ', theta.shape, type(theta))
print('Hypothesis: ', pred.shape)
print('Initial Cost to Minimize: ', cost)
print('Approx Gradient: ', grad)
# print('Optimal Parameters: ', opt_theta)