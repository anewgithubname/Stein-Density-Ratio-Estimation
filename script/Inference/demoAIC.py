from sdre.helper import *
from scipy.io import savemat
from sdre.estimators import dual

d = 8
n = 500

phi = lambda X: vstack([X, X**2])
f = lambda X,b:X[:b,:] 

def logpBar(x, theta):
    dimTheta = theta.shape[0]
    theta = theta.reshape([dimTheta,1])
    theta = vstack([theta, zeros([d - dimTheta, 1]), -1 / 2 * ones([d, 1])])
    return dot(theta.T, phi(x))

def infer(seed, dimTheta=1, b=1):
    random.seed(seed)
    print('seed:', seed)

    # Simple Gaussian dataset
    XData = random.standard_normal((d, n)) + array([[.1,.1,.1,.1,0,0,0,0]]).T

    # infer model parameters
    delta_dua,theta_dua,LL,TfData = dual(logpBar, lambda x:f(x,b), XData, theta=zeros(dimTheta))

    print(seed, 'log likelihood',LL)
    return LL - b + dimTheta

if __name__ == '__main__':
    s = zeros([d,d])
    for i in range(d):
        for j in range(d):
            if i<j:
                print(i+1,j+1)
                s[i,j] = infer(1,dimTheta = i+1, b= j+1)
                print(s[i,j])
    
    print(s)

    savemat('s.mat', {'s':s})