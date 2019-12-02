from sdre.helper import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from multiprocessing import Pool
from socket import gethostname
from time import time
from sdre.trainnn import lastlayer
from scipy.stats import multivariate_normal
from scipy import io as sio

optparam = load('optpara.npy',allow_pickle=True)
d = 784
n = 100
dimTheta = 20

f = lambda X: lastlayer(optparam,X.T).T

def logpBar(x, theta):
    theta = vstack([theta])
    return dot(theta.T,f(x))

gLogP = grad(logpBar)

def NCE(XData, logq, Xq, theta):
    mDn = XData.shape[1]/float(Xq.shape[1])
    r = lambda theta, X:exp(logpBar(X,theta[:-1,:]) - logq(X) - theta[-1,:])*mDn
    PPosGX = lambda theta, X: r(theta,X)/(1+r(theta,X)) 
    PNegGX = lambda theta, X: 1/(1+r(theta,X))
    
    #logistic regression
    return mean(log(PPosGX(theta, XData)))+mean(log(PNegGX(theta,Xq)))

def callbackF(Xi):
    global Nfeval
    print("iter {0:4d}, theta norm: {1:5.4f}".format(Nfeval, linalg.norm(Xi)))
    Nfeval += 1

def infer(seed, XData):
    random.seed(seed); print('seed:', seed)
    n0 = XData.shape[1]
    idx = random.permutation(n0)
    XData = XData[:,idx[:n]]
    
    # create the noise distribution
    Sigma = cov(XData) + .000001*eye(d)
    q = multivariate_normal(mean=mean(XData,1), cov = Sigma)
    Xq = q.rvs(XData.shape[1]*10).T
    U = linalg.cholesky(linalg.inv(Sigma))
    logq = lambda x: -sum(dot(U.T,x-mean(XData,1).reshape([d,1]))**2,0)/2
    obj = lambda theta: -NCE(XData,logq,Xq,array([theta]).T)

    grad_obj = grad(obj)
    hess_obj = jacobian(grad_obj)

    x0 = zeros(dimTheta+1)

    # t0 = time()
    # res = minimize(obj, x0, jac=grad_obj, callback=callbackF, method='CG',
    #                options={'disp': True, 'maxiter': 10000})
    # x0 = res.x
    # print('elapsed:', time() - t0)

    for i in range(5000):
        x0 = x0 -.1*grad_obj(x0)
        if i % 10 == 0:
            print("iter: {0:4d}, NCE obj:{1:4.4f}".format(i, obj(x0)))

    theta = x0
    print('estimate', theta)

    sio.savemat('out/nn %s %d.mat' % (gethostname(), seed),
                {'theta': theta[:-1], 'status': 0})
    return theta

if __name__ == '__main__':
    print("Loading training data...")
    train_images, train_labels, test_images, test_labels = loadMNIST(False)
    train_imagesO, train_labelsO, test_imagesO, test_labelsO = loadMNIST(False)

    for digit in range(10):
        Nfeval = 1
        XData = []; 
        for i in range(train_images.shape[0]):
            if train_labels[i,digit]==1:
                XData.append(train_images[i,:])
        XData = vstack(XData).T

        infer(digit,XData)

        theta = sio.loadmat('out/nn %s %d.mat' % (gethostname(), digit))['theta']
        XData = []
        for i in range(test_images.shape[0]):
            if test_labels[i,digit]==1:
                XData.append(test_images[i,:])
        XData = vstack(XData).T
        ll = logpBar(XData, theta.T)

        XData0 = []
        for i in range(test_imagesO.shape[0]):
            if test_labels[i,digit]==1:
                XData0.append(test_imagesO[i,:])
        XData0 = vstack(XData0).T
        sio.savemat('out/mnistD_NCE %d.mat' % digit,{'theta':theta,'ll':ll,'XData':XData,'XData0':XData0})
