import sdre
from sdre.helper import *

from numpy.linalg import inv
from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
from scipy.optimize import minimize, Bounds, NonlinearConstraint

def primal(logpBar, f, XData, solver="", 
           eta = .01, max_iter = 100000, tol = 1e-6, prt_interval = 100):
    d,n = XData.shape
    b = f(XData).shape[0]

    grad_logp = grad(logpBar)

    gFData = gradient_F(f, XData)
    TthetaF = steinFea(XData, gFData, grad_logp(XData), f, b)

    # Primal objective function
    objective = lambda para: -mean(log(dot(para.T, TthetaF) + 1),1)
    grad_obj = grad(objective)

    delta_init = zeros([b,1])

    if solver == "builtin":
        # # scipy build in minimization, not very stable, do not use
        res = minimize(lambda para:objective(para.reshape([b,1]))[0], 
                       jac=lambda para:grad_obj(para.reshape([b,1]))[:,0],
                       x0=delta_init[:,0],method='CG')
        print(res)
        delta_hat = res.x
    else:
        # gradient descent
        delta = delta_init
        delta_old = array([[inf]])
        for i in range(max_iter):
            delta = delta - eta*grad_obj(delta)
            if linalg.norm(delta_old - delta) >= tol:
                if i % prt_interval ==0:
                    print(i, 'delta:', delta[:,0])
                delta_old = delta
            else: 
                print('gradient descent converged after %d iterations' % i)
                break
        delta_hat = delta

    return delta_hat

def dual(logpBar, f, XData):
    d,n = XData.shape
    b = f(XData).shape[0]

    grad_logp = grad(logpBar)

    gFData = gradient_F(f, XData)
    TthetaF = steinFea(XData, gFData, grad_logp(XData), f, b)

    # Dual objective function
    obj = lambda para: mean(-log(-para[:n]) - 1) - mean(para[:n])
    grad_obj = grad(obj)
    
    # constraint
    nonlinC = lambda para: - dot(TthetaF, para[:n])
    jac_nonlinC = jacobian(nonlinC)

    # use the builtin optimizer to optimize dual
    NC = NonlinearConstraint(nonlinC, zeros(b), zeros(b), jac=jac_nonlinC)
    x0 = hstack([-ones(n), zeros(b)])

    res = minimize(obj, x0, jac=grad_obj, constraints=NC, method='trust-constr',
                   options={'disp': True, 'maxiter': 10000})
    
    # Solve a least square to get delta     
    r_n = -1. / res.x[:n]
    return inv(TthetaF.dot(TthetaF.T)).dot(TthetaF).dot(r_n-1)