import sdre
from sdre.helper import *

from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
from scipy.optimize import minimize, Bounds, NonlinearConstraint

def estimation(logpBar, f, XData, solver="builtin"):
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
        # # scipy build in minimization, not very stable
        res = minimize(lambda para:objective(para.reshape([b,1])),delta_init)
        delta_hat = res.x
    else:
        # gradient descent
        delta = delta_init
        delta_old = array([[inf]])
        eta = .01  #learning rate
        for i in range(10000):
            delta = delta - eta*grad_obj(delta)
            if linalg.norm(delta_old - delta) >= 1e-6:
                print('delta:', delta[:,0])
                delta_old = delta
            else: 
                print('gradient descent converged after %d iterations' % i)
                break
        delta_hat = delta

    return delta_hat