import numpy as np
from scipy.optimize import fmin_bfgs


# Rosenbrock Function
def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


# Broyden-Fletcher-Goldfarb-Shanno Algorithm Gradient (fmin_bfgs)
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der


x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
xopt = fmin_bfgs(rosen, x0, fprime=rosen_der)
print(xopt)


def optimize(self, min_func='minimize'):
    # Tuple of training examples
    my_args = (self.X_train, self.y_train)

    # Gradient Descent
    min_theta = minimize(fun=self.cost_function,
                         x0=self.theta,
                         args=my_args,
                         method='TNC',
                         jac=self.compute_grad)

    # BFGS algorithm
    bfgs_theta = fmin_bfgs(self.cost_function,
                           x0=self.theta,
                           args=my_args,
                           maxiter=400,
                           fprime=self.compute_grad)

    # Return minimized theta
    if min_func == 'bfgs':
        return bfgs_theta.x
    else:
        return min_theta.x

