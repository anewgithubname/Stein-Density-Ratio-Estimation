#%% 
import sdre
from sdre.helper import *
from sdre.estimators import primal
from sdre.estimators import dual

from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from matplotlib import pyplot as plt

d = 1
n = 100

# Dataset
random.seed(1)
XData = random.standard_normal((d, n))

#%%
# Unnormalized density
logpBar = lambda x:-sum(x**2 + ones([d,1])*x, 0) / 2
# f in Stein feature
f = lambda X: vstack([X,X**2/2,X**3/3])
b = 3*d

grad_logp = grad(logpBar)

#%% 
# Estimate density ratio parameters
delta_pri = primal(logpBar, f, XData, eta = .001, max_iter=50000)
delta_dua,theta_dua = dual(logpBar, f, XData)

#%% plotting
r_pri = lambda x: delta_pri.T.dot(steinFea(x, traceHessF(f, x), grad_logp(x), f, b)) + 1
r_dua = lambda x: delta_dua.T.dot(steinFea(x, traceHessF(f, x), grad_logp(x), f, b)) + 1

xMarks = array([linspace(-5,5)])
plt.plot(xMarks[0,:],r_pri(xMarks)[0,:],c='r',
label='r_primal')
plt.plot(xMarks[0,:],r_dua(xMarks),c='b',
label='r_dual')

plt.scatter(XData, r_dua(XData),c='b',
marker='x', label='r_dual(Data)')

ax = plt.gca()
ax.set_xlim([min(XData)-.5,max(XData)+.5])
ax.set_ylim([0,max(r_pri(XData))+.5])
ax.legend()
plt.show()

