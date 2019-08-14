import sys; sys.path.insert(1, 'util/')
from helper import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint
# from socket import gethostname
from time import time

# dimension, num of samples, dimension of theta
d = 5
n = 500
dimTheta = d

# construct feature function f
f = lambda X: vstack([X,X**2])
b = dimTheta*2

# log P(x;theta) model, unnormalized
def logpBar(x, theta):
    return -sum(x**2 / 2- theta*x[:dimTheta,:], 0)

grad_logp = grad(logpBar)
# Dual objective function
obj = lambda para: mean(-log(-para[:n]) - 1) - mean(para[:n])
grad_obj = grad(obj)

# theta: parameter of the unormalized log-density function, XData: dataset
def trial(theta, XData):
    n0 = XData.shape[1]
    idx = random.permutation(n0)
    XData = XData[:,idx[:n]]

    gFData = gradient_F(f, XData)
    TthetaF = lambda theta: steinFea(XData, gFData, grad_logp(XData, theta), f, b)

    # constraint
    nonlinC = lambda para: - \
        dot(TthetaF(para[n:].reshape([dimTheta, 1])), para[:n])
    jac_nonlinC = jacobian(nonlinC)

    # use the builtin optimizer to optimize dual
    NC = NonlinearConstraint(nonlinC, zeros(b), zeros(b), jac=jac_nonlinC)
    x0 = hstack([-ones(n), zeros(dimTheta)])

    t0 = time()
    res = minimize(obj, x0, jac=grad_obj, constraints=NC, method='trust-constr',
                   options={'disp': True, 'maxiter': 10000})
    print('elapsed:', time() - t0)
    theta = res.x[n:]
    print('theta_hat', theta)
    mu = res.x[:n]
    print('\noptimizaiton result', res.status)
    if res.status < 0:
        return -1
    print('obj value (prim vs. dual)', 2 *
          sum(log(-1. / res.x[:n])), 2 * n * obj(res.x), '\n')
    return theta

if __name__ == '__main__':
    random.seed(1)
    # Dataset
    theta_star = tile(pi,[5,1])
    XData = random.standard_normal((d, n)) + theta_star
    # Run the main programme
    theta_hat = trial(-ones([d,1])*.2, XData)
    print('theta_hat', theta_hat)
    print('theta_star', theta_star.flatten())
