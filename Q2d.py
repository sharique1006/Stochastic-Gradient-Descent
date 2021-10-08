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

# 2. Sampling and Stochastic Gradient

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

	return thetaL, itr

thetaL2, maxit2 = stochasticGradientDescent(X, Y, 100)
thetaL3, maxit3 = stochasticGradientDescent(X, Y, 10000)

# (d) 3D Plot of Parameters

def plotTheta(thetaL, r):
	thetaL = np.array(thetaL)
	thetaL = thetaL.reshape(-1, 3)
	ax = plt.axes(projection='3d')
	x = [] 
	y = [] 
	z = []
	for i in range(len(thetaL)):
		x.append(thetaL[i][0])
		y.append(thetaL[i][1])
		z.append(thetaL[i][2])

	ax.scatter(x, y, z, color='red')
	ax.set_xlabel(r'$\theta_0$')
	ax.set_ylabel(r'$\theta_1$')
	ax.set_zlabel(r'$\theta_2$')
	ax.set_title("Parameter Plot - " + r'$r = {0}$'.format(r))
	f = "Q2d3DPlot" + str(r) + ".png" 
	parameter_plot3d = os.path.join(sys.argv[2], f)
	plt.savefig(parameter_plot3d)
	plt.close()

plotTheta(thetaL2, 100)
plotTheta(thetaL3, 10000)