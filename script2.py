import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD

    classes = np.unique(y).astype(int)
    d = np.shape(X)[1]
    means = np.zeros((d, classes.size));
    covmat = np.zeros((d, d))
    for i in classes:
        trainDataX = X[np.where(y == i)[0],:]
        means[:, i-1] = np.mean(trainDataX, 0).T
        covmat = covmat + np.cov(trainDataX.T)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    classes = np.unique(y).astype(int)
    d = np.shape(X)[1]
    means = np.zeros((d, classes.size));
    covmats = []
    for i in classes:
        trainDataX = X[np.where(y==i)[0],:]
        means[:, i-1] = np.mean(trainDataX, 0).T
        covmats.append(np.cov(trainDataX.T))
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    classes = np.unique(y).astype(int)
    predicted = np.zeros((np.shape(Xtest)[0],1))
    divider = np.sqrt(np.pi*2 * np.linalg.det(covmat))
    correct = 0.0;
    for i in range(0, Xtest.shape[0]):
        oldpdf = 0;
        for j in classes:
            exp = np.exp((-0.5)*(np.dot(np.dot((Xtest[i,:].T - means[:, j-1]).T, np.linalg.inv(covmat)), (Xtest[i,:].T - means[:, j-1]).T)))
            newpdf = exp/divider
            if newpdf > oldpdf:
                oldpdf = newpdf
                predictedClass = j
        predicted[i] = predictedClass

    for i in range(0, predicted.shape[0]):
        if predicted[i] == ytest[i]:
            correct = correct + 1
    acc = (correct / Xtest.shape[0]) * 100
    ypred = predicted.T
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    classes = np.unique(y).astype(int)
    predicted = np.zeros((np.shape(Xtest)[0],1))
    divider = np.zeros(classes.size)
    correct = 0.0
    for i in classes:
        D = np.shape(covmats[i-1])[0]
        divider[i-1] = np.power(2*np.pi, D/2)*np.sqrt(np.linalg.det(covmats[i-1]))

    for i in range (0, np.shape(Xtest)[0]):
        oldpdf = 0
        for j in classes:
            exp = np.exp((-0.5)*(np.dot(np.dot((Xtest[i,:].T - means[:, j-1]).T, np.linalg.inv(covmats[j-1])), (Xtest[i,:].T - means[:, j-1]).T)))
            newpdf = exp / divider[j - 1]
            if newpdf > oldpdf:
                oldpdf = newpdf
                predictedClass = j
        predicted[i] = predictedClass

    for i in range(0, predicted.shape[0]):
        if predicted[i] == ytest[i]:
            correct = correct + 1;
    acc = (correct / np.shape(Xtest)[0]) * 100
    ypred = predicted.T
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD

    xt_x = np.dot(X.T,X)
    xt_y = np.dot(X.T,y)

    #calculating inverse using solve
    w = np.linalg.solve(xt_x,xt_y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD

    Id = np.identity(X.shape[1])

    xt_x = np.dot(X.T,X)
    xt_y = np.dot(X.T,y)

    w = np.linalg.solve((lambd*Id + xt_x),xt_y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD

    rmse = 0
    N = Xtest.shape[0]
    for i in range(0,N):
        y_i = ytest[i]
        wt  = w.T
        x_i = Xtest[i]
        rmse = rmse + np.square(y_i - np.dot(wt,x_i))

    rmse = np.sqrt(rmse/N)
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD

    w = w.reshape(-1,1)

    wt = w.T
    xt = X.T

    xw = np.dot(X,w)

    y_xw = (y - xw)
    y_xwt = np.transpose(y_xw)

    tmp1 = np.dot(y_xwt,y_xw)
    tmp2 = lambd * np.dot(wt,w)

    error = (tmp1 + tmp2)/2

    xtx = np.dot(xt,X)
    xty = np.dot(xt,y)
    xtxw = np.dot(xtx,w)

    error_grad = xtxw - xty + lambd*w
    error_grad = error_grad.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD

    Xd = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xd[:, i] = x ** i
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc, lPred = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc, qPred = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
# plt.show()
# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
# plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
# plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
# plt.show()