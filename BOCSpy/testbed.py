from itertools import combinations
import numpy as np

def order_effects(x_vals, ord_t):
        # order_effects: Function computes data matrix for all coupling
        # orders to be added into linear regression model.

        # Find number of variables
        n_samp, n_vars = x_vals.shape

        # Generate matrix to store results
        x_allpairs = x_vals

        for ord_i in range(2,ord_t+1):

            # generate all combinations of indices (without diagonals)
            offdProd = np.array(list(combinations(np.arange(n_vars),ord_i)))

            # generate products of input variables
            x_comb = np.zeros((n_samp, offdProd.shape[0], ord_i))
            for j in range(ord_i):
                x_comb[:,:,j] = x_vals[:,offdProd[:,j]]
            x_allpairs = np.append(x_allpairs, np.prod(x_comb,axis=2),axis=1)

        return x_allpairs

order_effects(x_vals, 1)

# -----

from LinReg import LinReg
import bhs
import importlib
importlib.reload(bhs)
from bhs import bhs, standardise, fastmvg_rue, fastmvg

LR = LinReg(8, 2)

# set data
LR.xTrain = inputs['x_vals']
LR.yTrain = inputs['y_vals']

# setup data for training
LR.setupData()

# create matrix with all covariates based on order
LR.xTrain = LR.order_effects(LR.xTrain, LR.order)

bhs(LR.xTrain, LR.yTrain, int(1e3), 0, 1)

# bhs
# ===

n, p = LR.xTrain.shape
X, _, _, y, muY = standardise(LR.xTrain, LR.yTrain)
X - LR.xTrain

XtX = np.matmul(X.T,X)
sigma2  = 1.
lambda2 = np.random.uniform(size=p)
tau2    = 1.
nu      = np.ones(p)
xi      = 1.

sigma = np.sqrt(sigma2)
Lambda_star = tau2 * np.diag(lambda2)
b = fastmvg_rue(X/sigma, XtX/sigma2, y/sigma, sigma2*Lambda_star)

# Sample sigma2
e = y - np.dot(X,b)
shape = (n + p) / 2.
scale = np.dot(e.T,e)/2. + np.sum(b**2/lambda2)/tau2/2.
sigma2 = 1. / np.random.gamma(shape, 1./scale)

if len(b.shape) == 2 and b.shape[1] == 1:
    b.shape = (len(b),)

# Sample lambda2
scale = 1./nu + b**2./2./tau2/sigma2
lambda2 = 1. / np.random.exponential(1./scale)

# fastmvg_rue
# ===========

Phi, PtP, alpha, D = X/sigma, XtX/sigma2, y/sigma, sigma2*Lambda_star

p = Phi.shape[1] # 36
Dinv = np.diag(1./np.diag(D)) # 36 * 36

# regularize PtP + Dinv matrix for small negative eigenvalues
try:
    L = np.linalg.cholesky(PtP + Dinv)
except:
    mat  = PtP + Dinv
    Smat = (mat + mat.T)/2.
    maxEig_Smat = np.max(np.linalg.eigvals(Smat))
    L = np.linalg.cholesky(Smat + maxEig_Smat*1e-15*np.eye(Smat.shape[0]))

v = np.linalg.solve(L, np.dot(Phi.T,alpha))
m = np.linalg.solve(L.T, v)
w = np.linalg.solve(L.T, np.random.randn(p))
w.shape = (len(w), 1)

x = m + w

# fastmvg

d = np.diag(D)
u = np.random.randn(p) * np.sqrt(d)
delta = np.random.randn(n)
v = np.dot(Phi,u) + delta
#w = np.linalg.solve(np.matmul(np.matmul(Phi,D),Phi.T) + np.eye(n), alpha - v)
#x = u + np.dot(D,np.dot(Phi.T,w))
mult_vector = np.vectorize(np.multiply)
Dpt = mult_vector(Phi.T, d[:,np.newaxis])
w = np.linalg.solve(np.matmul(Phi,Dpt) + np.eye(n), alpha - v)
x = u + np.dot(Dpt,w)

# ---

a = np.array([[1, 2], [3, 5]])
b = np.array([1, 2])
b.shape = (2, 1)
x = np.linalg.solve(a, b)
x