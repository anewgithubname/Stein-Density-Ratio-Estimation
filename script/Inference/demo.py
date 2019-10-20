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
# Unnormalized density parameterized by theta
logpBar = lambda x,theta:-sum(x**2 + theta*ones([d,1])*x, 0) / 2
# f in Stein feature
f = lambda X: vstack([X,X**2/2,X**3/3])
b = 3*d

#%% 
# infer model parameters
delta_dua,theta_dua,LL = dual(logpBar, f, XData, theta=1)

#%% 
print('ground truth:', 0, 'estimated', theta_dua[0])
print('log likelihood', LL)