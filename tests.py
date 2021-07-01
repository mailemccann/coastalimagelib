import numpy as np

x = [5,25]
xvec = np.arange(x[0], x[1]+1, 1)
yvec = [10]
X, Y = np.meshgrid(xvec, yvec)
print(X)
print(Y)