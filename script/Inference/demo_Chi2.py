#%% 
import sdre
from sdre.helper import *
from sdre.estimators import dual
from scipy import stats
from scipy import io

from autograd.numpy import *
from matplotlib import pyplot as plt

from multiprocessing import Pool
from time import time

d = 20
n = 250
dTheta = d-5

#%%
# Unnormalized density parameterized by theta
logpBar = lambda x,theta:-sum(x**2, 0) / 2 + \
            sum(reshape(theta,[dTheta,1])*ones([dTheta,1])*tanh(x[:dTheta,:]),0)
# f in Stein feature
f = lambda X: vstack([tanh(X)])
b = d

#%% Show asymptotic distribution of Log-likelihood
def infer(seed):   
    # Generate Dataset
    random.seed(seed)
    XData = random.standard_normal((d, n))
    # infer model parameters
    delta_dua,theta_dua,LL,TfData = dual(logpBar, f, XData, theta=zeros(dTheta))

    print(seed, 'log likelihood',LL)
    return LL

if __name__ == '__main__':
    # LL = infer(0)
    # LL = 2*array(list(map(infer, range(20))))

    start = time()
    # if you want to be parallel...
    ncores = getThreads(); print('number of threads,', ncores)
    LL = 2*array(list(Pool(ncores).map(infer, range(10000))))
    print('elapsed:', time() - start)
    
    # output results to matlab file
    io.savemat('LL.mat',{'LL':LL}) 

    # plot results    
    res = stats.probplot(LL,dist=stats.chi2, sparams=(d-dTheta,), plot=plt)
    plt.show()