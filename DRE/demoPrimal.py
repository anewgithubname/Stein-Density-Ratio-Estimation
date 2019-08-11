import sys; sys.path.insert(1, 'util/')
from helper import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from socket import gethostname
from time import time

# dimension, num of samples, dimension of theta
d = 5
n = 500
dimTheta = d

# construct feature function f
f = lambda X: X
b = dimTheta

# log P(x;theta) model, unnormalized
def logpBar(x, theta):
    return -sum(x**2- theta*x[:dimTheta,:], 0) / 2

# Assemble Stein Feature
def steinFea(X, grad_f, grad_logp):
    fea = []
    for i in range(b):
        t = grad(lambda X: f(X)[i,:])
        fea.append(sum(grad_logp * t(X),0) + grad_f[i,:])
    return vstack(fea)

grad_logp = grad(logpBar)

# theta: parameter of the unormalized log-density function, XData: dataset
def trial(theta, XData):
    n0 = XData.shape[1]
    idx = random.permutation(n0)
    XData = XData[:,idx[:n]]

    gFData = gradient_F(f, XData)
    TthetaF = steinFea(XData, gFData, grad_logp(XData, theta))

    # Primal objective function
    objective = lambda para: -mean(log(dot(para.T, TthetaF) + 1),1)
    grad_obj = grad(objective)

    delta_init = zeros([d,1])
    # scipy build in minimization, not very stable
    res = minimize(lambda para:objective(para.reshape([d,1])),delta_init)
    delta_hat1 = res.x

    # gradient descent
    delta = delta_init
    delta_old = array([[inf]])
    eta = .1  #learning rate
    for i in range(1000):
        delta = delta - eta*grad_obj(delta)
        if linalg.norm(delta_old - delta) >= 1e-6:
            print('delta:', delta[:,0])
            delta_old = delta
        else: 
            print('gradient descent converged after %d iterations' % i)
            break
    delta_hat2 = delta

    return delta_hat1, delta_hat2

if __name__ == '__main__':
    random.seed(1)
    # Dataset
    XData = random.standard_normal((d, n))
    # Run the main programme
    d1, d2 = trial(-ones([d,1])*.2, XData)
    print('\n diff between GD solver and builtin solver:')
    print(d1 - d2[:,0])