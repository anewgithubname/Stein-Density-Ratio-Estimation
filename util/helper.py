from autograd.numpy import *
from autograd import elementwise_grad as grad
from autograd import jacobian as jacobian
# import scipy.io as sio
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

# Gradient of Feature Function
def gradient_F(K, X):
    d = X.shape[0]; b = K(X).shape[0]
    grad2_K = []
    for i in range(b):
        t = grad(lambda X: K(X)[i,:])

        grad2_Ki = 0
        for j in range(d):
            t2 = grad(lambda X: t(X)[j,:])
            grad2_Ki = grad2_Ki + t2(X)[j,:]
        grad2_K.append(grad2_Ki)
        print(i)
    return vstack(grad2_K)

# Assemble Stein Feature
def steinFea(X, grad_f, grad_logp, f, b):
    fea = []
    for i in range(b):
        t = grad(lambda X: f(X)[i,:])
        fea.append(sum(grad_logp * t(X),0) + grad_f[i,:])
    return vstack(fea)
