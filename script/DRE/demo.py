import sdre
from sdre.helper import *
from sdre.dre import estimation

from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from matplotlib import pyplot as plt

d = 1
n = 1000

# Dataset
XData = random.standard_normal((d, n))

# Unnormalized density
logpBar = lambda x:-sum(x**2- .5*ones([d,1])*x, 0) / 2
# f in Stein feature
f = lambda X: vstack([X,X**2,X**3])
b = 3*d

# Estimate density ratio parameters
delta = estimation(logpBar, f, XData, solver="")

grad_logp = grad(logpBar)
print(delta)

# estimated ratio function
r = lambda x: delta.T.dot(steinFea(x, gradient_F(f, x), grad_logp(x), f, b)) + 1
xMarks = array([linspace(-2.5,3.5)])

print(r(xMarks))
plt.plot(xMarks[0,:],r(xMarks)[0,:])
plt.show()

