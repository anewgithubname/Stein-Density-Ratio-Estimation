import sdre
from sdre.helper import *
from sdre.dre import estimation

from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
from scipy.optimize import minimize, Bounds, NonlinearConstraint

d = 5
n = 500

# Dataset
XData = random.standard_normal((d, n))

# Unnormalized density
logpBar = lambda x:-sum(x**2- ones([d,1])*.2*x, 0) / 2
# f in Stein feature
f = lambda X: X

# Estimate density ratio parameters
delta = estimation(logpBar, f, XData)
print(delta)