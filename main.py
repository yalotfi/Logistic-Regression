import logistic_regression as lr
import process_csv as data


file_path = 'grades.txt'
[X_train, X_test,
y_train, y_test, theta] = data.process_csv(file_path, 0)

# theta = np.matrix([[0 for param in range(n + 1)]])
reg_lambda = 0

# Build and Train Model
lrGrades = lr.BasicLogistic(X_train, y_train, theta, reg_lambda)
min_theta = lrGrades.optimize(min_func='minimize')
cost = lrGrades.cost_function(min_theta, X_train, y_train)

# Assess Model
print('X_train: ', X_train.shape, type(X_train))
print('y_train: ', y_train.shape, type(y_train))
print('Theta: ', theta.shape, type(theta))
print('Optimal Parameters: ', min_theta)
print('Min_Cost: ', cost)
