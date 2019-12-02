from sdre.helper import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from multiprocessing import Pool
from socket import gethostname
from time import time
from sdre.trainnn import lastlayer
from kgof.kernel import KGauss
import kgof.util as util
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

def KSD(XData, k1, k2, k3, k4, theta):
    g = gLogP(XData, theta)

    t2 = zeros([n,n])
    for i in range(d):
        t2 = t2 + k2[i]*g[i,:]
    
    t3 = zeros([n,n])
    for i in range(d):
        t3 = t3 + (k3[i].T*g[i,:]).T

    return sum((dot(g.T,g)*k1 + t2 + t3 + k4).flatten())/n**2

def callbackF(Xi):
    global Nfeval
    print("iter {0:4d}, theta norm: {1:5.4f}".format(Nfeval, linalg.norm(Xi)))
    Nfeval += 1

def infer(digit, XData):
    random.seed(1); print('digit:', digit)
    n0 = XData.shape[1]
    idx = random.permutation(n0)
    XData = XData[:,idx[:n]]

    sig2 = util.meddistance(XData.T, subsample=1000)**2

    kG = KGauss(sig2)
    k1 = kG.eval(XData.T,XData.T)
    k4 = kG.gradXY_sum(XData.T,XData.T)

    k2 = []
    for i in range(d):
        k2.append(kG.gradX_Y(XData.T, XData.T,i))
    
    k3 = []
    for i in range(d):
        k3.append(kG.gradY_X(XData.T, XData.T,i))

    obj = lambda theta:KSD(XData, k1, k2, k3, k4, array([theta]).T)
    grad_obj = grad(obj)
    hess_obj = jacobian(grad_obj)

    x0 = random.randn(dimTheta)
    t0 = time()
    res = minimize(obj, x0, jac=grad_obj, method='BFGS',callback=callbackF, 
                   options={'disp': True, 'maxiter': 10000})
    print('elapsed:', time() - t0)

    theta = res.x
    print('estimate', theta)
    print('\noptimizaiton result', res.status)
    if res.status < 0:
        return -1

    sio.savemat('out/nn %s %d.mat' % (gethostname(), digit),
                {'theta': theta, 'status': res.status})
    return obj(res.x)

if __name__ == '__main__':
    print("Loading training data...")
    test_images, test_labels, train_images,  train_labels = loadMNIST(False)
    test_imagesO, test_labelsO, train_imagesO,  train_labelsO = loadMNIST(False)

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
        sio.savemat('out/mnistD_KSD %d.mat' % digit,{'theta':theta,'ll':ll,'XData':XData,'XData0':XData0})
