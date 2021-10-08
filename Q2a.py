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