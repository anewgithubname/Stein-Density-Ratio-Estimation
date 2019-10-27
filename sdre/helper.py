from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
from sdre.data import load_mnist
import sys,os

set_printoptions(precision=8)

def kernel_gauss(X, Y, sigma2):
    sumx2 = reshape(sum(X**2, 0),[-1,1])
    sumy2 = reshape(sum(Y**2, 0).T,[1,-1])
    D2 = sumx2 - 2*dot(X.T, Y) + sumy2
    return exp(-D2/(2.0*sigma2))

def kernel_poly(X):
    d, n = X.shape
    t = [X, ones([1,n])]
    for i in range(d):
        for j in range(d):
            if i<= j:
                t.append(X[i,:]*X[j,:])
    return vstack(t)

# trace of Hessian of F
def traceHessF(F, X):
    d = X.shape[0]; b = F(X).shape[0]
    grad2_F = []
    for i in range(b):
        t = grad(lambda X: F(X)[i,:])

        grad2_Fi = 0
        for j in range(d):
            t2 = grad(lambda X: t(X)[j,:])
            grad2_Fi = grad2_Fi + t2(X)[j,:]
        grad2_F.append(grad2_Fi)
        # print(i)
    return vstack(grad2_F)

# Assemble Stein Feature
def steinFea(X, traceHessF, grad_logp, f, b):
    fea = []
    for i in range(b):
        t = grad(lambda X: f(X)[i,:])
        fea.append(sum(grad_logp * t(X),0) + traceHessF[i,:])
    return vstack(fea)

#Load mnist
def loadMNIST(normalize = False):
    N, train_images, train_labels, test_images,  test_labels = load_mnist()
    if normalize:
        mu = mean(train_images,0)
        std0 = std(train_images,0)
        train_images = train_images - mu
        test_images = test_images - mu
        train_images = train_images/std0
        test_images = test_images/std0
        train_images[isnan(train_images)] = 0
        test_images[isnan(test_images)] = 0
    return train_images, train_labels, test_images, test_labels

def getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c cores /proc/cpuinfo').read())