import numpy as np
import matplotlib.pyplot as plt
  
y = lambda x: (6 * np.cos(1) - 3) * np.cos(x) / np.sin(1) + 6 * np.sin(x) + x ** 3 - 6 * x
dy = lambda x: (3 - 6 * np.cos(1)) * np.sin(x) / np.sin(1) + 6 * np.cos(x) + 3 * x ** 2 - 6
#%%
x = np.linspace(0, 1, 100)
h = x[1] - x[2]
h = 1
n = 7
A = np.zeros((n, n))
for i in range(2, n - 2):
    A[i, i - 2] = -1 / (12 * h ** 2)
    A[i, i - 1] = 16 / (12 * h ** 2)
    A[i, i] = - 30 / (12 * h ** 2)
    A[i, i + 1] = 16 / (12 * h ** 2)
    A[i, i + 2] = -1 / (12 * h ** 2) 
print(A)
#%%
plt.plot(x, y(x))
plt.plot(x, dy(x))
plt.grid(True)
plt.show()