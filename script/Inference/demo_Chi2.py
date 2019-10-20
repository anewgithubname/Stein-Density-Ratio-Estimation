#%% 
import sdre
from sdre.helper import *
from sdre.estimators import primal
from sdre.estimators import dual
from scipy import stats
from scipy import io

from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from matplotlib import pyplot as plt

from multiprocessing import Pool

d = 20
n = 500
dTheta = d-1

#%%
# Unnormalized density parameterized by theta
logpBar = lambda x,theta:-sum(x**2, 0) / 2 + \
            sum(reshape(theta,[dTheta,1])*ones([dTheta,1])*x[:dTheta,:],0)
# f in Stein feature
f = lambda X: vstack([X])
b = d

#%% Show asymptotic distribution of Log-likelihood
def infer(seed):   
    # Generate Dataset
    random.seed(seed)
    XData = random.standard_normal((d, n))
    # infer model parameters
    delta_dua,theta_dua,LL = dual(logpBar, f, XData, theta=zeros(dTheta))

    print(seed, 'log likelihood',LL)
    return LL

if __name__ == '__main__':
    # infer(0)
    # LL = list(map(infer, range(1000)))


    LL = 2*array(list(Pool(12).map(infer, range(10000))))
#%%
# rvs = stats.chi2.rvs(2,size=100)
# res = stats.probplot(LL,dist=stats.chi2, sparams=(2,), plot=plt)
# n, bins, patches = plt.hist(LL, 50, density=True, facecolor='g')
    io.savemat('LL.mat',{'LL':LL})