from sdre.helper import *
from multiprocessing import Pool
from socket import gethostname
from time import time
from sdre.trainnn import lastlayer
from sdre.estimators import dual
from scipy import io as sio

optparam = load('optpara.npy',allow_pickle=True)
d = 784
n = 100
dimTheta = 20

f = lambda X: vstack([lastlayer(optparam,X.T).T])
b = dimTheta

def logpBar(x, theta):
    theta = vstack([theta])
    return dot(reshape(theta,[dimTheta,1]).T,f(x))

def infer(digit, XData):
    random.seed(1); print('digit:', digit)
    n0 = XData.shape[1]
    idx = random.permutation(n0)
    XData = XData[:,idx[:n]]
    
    t0 = time()
    delta,theta,LL,TthetaF = dual(logpBar, f, XData, theta=zeros(dimTheta))
    print('elapsed:', time() - t0)
    print('estimate', theta)
    
    sio.savemat('out/nn %s %d.mat' % (gethostname(), digit),
                {'theta': theta, 'Ttheta': TthetaF})
    return 2 * n * LL

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
        # XData = random.standard_normal((d, n))
        ll = logpBar(XData, theta.T)

        XData0 = []
        for i in range(test_imagesO.shape[0]):
            if test_labels[i,digit]==1:
                XData0.append(test_imagesO[i,:])
        XData0 = vstack(XData0).T
        sio.savemat('out/mnistD_DLE %d.mat' % digit,{'theta':theta,'ll':ll,'XData':XData,'XData0':XData0})