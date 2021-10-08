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
out = os.path.join(sys.argv[2], 'Q2b.txt')
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
	print('\nBatch Size (r) = {}'.format(r), file=outfile)
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

		if r == m and itr % 1000 == 0:
			print('iteration {}: error = {} '.format(itr, error), end = '', file=outfile)
			print('w = {0},{1},{2}'.format(theta[0], theta[1], theta[2]), file=outfile)
		elif r != m:
			print('iteration {}: error = {} '.format(itr, error), end = '', file=outfile)
			print('w = {0},{1},{2}'.format(theta[0], theta[1], theta[2]), file=outfile)
	print("Stopping Criteria: Epoch > 7 and Error < 1e-7", file=outfile)
	print("Max Iterations = ", itr, file=outfile)
	print("Error =", error, file=outfile)
	print("Final Parameters = {0},{1},{2}\n".format(theta[0], theta[1], theta[2]), file=outfile)
	return thetaL, itr

thetaL1, maxit1 = stochasticGradientDescent(X, Y, 1)
thetaL2, maxit2 = stochasticGradientDescent(X, Y, 100)
thetaL3, maxit3 = stochasticGradientDescent(X, Y, 10000)
thetaL4, maxit4 = stochasticGradientDescent(X, Y, 1000000)