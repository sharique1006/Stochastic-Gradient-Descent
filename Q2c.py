import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import os

data_dir = sys.argv[1]
out_dir = sys.argv[2]
test_set = os.path.join(sys.argv[1], 'q2test.csv')
out = os.path.join(sys.argv[2], 'Q2c.txt')
outfile = open(out, "w")

# 2. Sampling and Stochastic Gradient
print("\n################ 2. Sampling and Stochastic Gradient ################", file=outfile)

# (a) Sampling

w = np.array([[3],[1],[2]])
X1 = np.random.normal(3, 2, (int)(1e6))
X2 = np.random.normal(-1, 2, (int)(1e6))
noise = np.random.normal(0, np.sqrt(2), int(1e6)).reshape(-1,1)

X = np.column_stack((X1, X2))
X = np.column_stack((np.ones(X.shape[0]), X))
Y = np.dot(X, w) + noise
m = len(Y)


# (b) Stochastic Gradient

def hw(theta, i, r):
	x = X[i*r:(i+1)*r,]
	return np.dot(x, theta)

def Jw(h, i, r):
	y = Y[i*r:(i+1)*r,]
	return (0.5/r)*np.sum((y - h)**2)

def dJw(h, i, r):
	x = X[i*r:(i+1)*r,]
	y = Y[i*r:(i+1)*r,]
	return (1.0/r)*np.dot(x.T, (y - h))

def stochasticGradientDescent(x, y, r):
	b = (int)(m/r)
	eta = 0.001
	theta = np.zeros((x.shape[1], 1))
	thetaL = [theta]
	prevCost = 0
	converged = False
	itr = 0
	while not converged:
		for i in range(b):
			h = hw(theta, i, r)
			cost = Jw(h, i, r)
			theta = theta + eta*dJw(h, i, r)
			thetaL.append(theta)
		itr += 1
		error = abs(cost - prevCost)
		prevCost = cost
		if error < 1e-7 or itr > 25000:
			converged = True

	return thetaL, itr, cost

thetaL1, maxit1, cost1 = stochasticGradientDescent(X, Y, 1)
thetaL2, maxit2, cost2 = stochasticGradientDescent(X, Y, 100)
thetaL3, maxit3, cost3 = stochasticGradientDescent(X, Y, 10000)
thetaL4, maxit4, cost4 = stochasticGradientDescent(X, Y, 1000000)

# (c) Test Data

data = np.loadtxt(test_set, delimiter=',', skiprows=1)
q2X1 = data[:,0]
q2X2 = data[:,1]
q2X = np.column_stack((q2X1, q2X2))
q2X = np.column_stack((np.ones(q2X.shape[0]), q2X))
q2Y = data[:,-1].reshape(-1,1)
q2m = len(q2Y)

def testError(theta):
	h = np.dot(q2X, theta)
	return (0.5/q2m)*np.sum((q2Y - h)**2)

error1 = testError(thetaL1[-1])
error2 = testError(thetaL2[-1])
error3 = testError(thetaL3[-1])
error4 = testError(thetaL4[-1])

print('\nFor r = 1 : Training Error = {}, Test Error = {}'.format(cost1, error1), file=outfile)
print('For r = 100 : Training Error = {}, Test Error = {}'.format(cost2, error2), file = outfile)
print('For r = 10000 : Training Error = {}, Test Error = {}'.format(cost3, error3), file=outfile)
print('For r = 1000000 : Training Error = {}, Test Error = {}'.format(cost4, error4), file=outfile)
