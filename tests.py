import numpy as np

Ud = np.ones((5,5))
Vd = np.ones((5,5))*2
# Initialize Flag
mask = np.ones_like(Ud)

# Flag negative UV coordinates
mask[(Ud < 2)] = 0
mask[(Vd < 2)] = 0

print(mask)