"""
  Question 4: Pade for EDOs
"""
import numpy as np
import matplotlib.pyplot as plt
import pathlib
  
#%%
ya = lambda x: (6 * np.cos(1) - 3) * np.cos(x) / np.sin(1) + 6 * np.sin(x) + x ** 3 - 6 * x # Analytic solution
f = lambda x: x ** 3 # RHS

# Solve system to get coefficients
def solveSystem():
  A1 = np.array([
      [1, 1, 1, 0, 0],
      [0, 1, 2, 1, 0],
      [0, 1, 4, -2, -2],
      [0, 1, 8, 3, -6],
      [0, 1, 16, -4, -12]
  ])
  
  A2 = np.array([
      [1, 1, 1, 0, 0],
      [0, -1, -2, 1, 0],
      [0, 1, 4, 2, -2],
      [0, -1, -8, 3, 6],
      [0, 1, 16, 4, -12]
  ])
  
  b1 = np.array([0, 0, 2, 0, 0])
  b2 = np.array([0, 0, 2, 0, 0])
  
  x1 = np.linalg.solve(A1, b1)
  x2 = np.linalg.solve(A2, b2)
  
  print("System coefficients")
  print(x1)
  print(x2)
  
# Solve system and show coefficients
solveSystem()  
  

def createMatrices(n, h):
  # A matrix
  A = np.zeros((n, n))
  np.fill_diagonal(A[1:], 1) # lower diagonal
  A += A.T # upper diagonal
  np.fill_diagonal(A, 10) # main diagonal
  A[0,:2] = np.array([1, -2.63636364]) # Coefficient b_2
  A[-1,-2:] = np.array([-2.63636364, 1]) # Coefficient d_2

  # B matrix
  B = np.zeros((n, n))
  np.fill_diagonal(B[1:], 12)
  B += B.T
  np.fill_diagonal(B, -24)

  B[0,:4] = np.array([-3., 5.45454545, -2.45454545, 0]) # Coefficients a_1, a_2, a_3, a_4=0 (boundary condition)
  B[-1,-4:] = np.array([0, -2.45454545, 5.45454545, -3]) # Coefficients c_1, c_2, c_3, c_4=0 (boundary condition)   
  B /= h ** 2 # 1/h^2 B

  return A, B
  
  
def experiment(n):
  # Domain discretization
  h = 1/(n+1) # \Delta x
  i = np.arange(0, n) # i for grid
  x = h * (i+1) # Fixed domain discretization by Pablo Huerta 
  
  A, B = createMatrices(n, h) # Create matrices A and B
  F = f(x) # RHS 
  I = np.eye(n) # I matrix for y vector
  
  # Solve EDO 
  y = np.linalg.solve(np.dot(np.linalg.inv(A), B) + I, F)
  
  return x, y

# Plot approximation and analytic
def plot(x, y, y_a):
  plt.plot(x, y, label="Real")
  plt.plot(x, y_a, label="Approximation")
  plt.grid(True)
  plt.legend()
  plt.show()

# Plot convergence
def plotConv(ns, err):
  plt.plot(ns, err, '-o', label="Error")
  # Reference convergence curves
  plt.plot(ns, 1/ns**3, label=r"$O(h^3)$")
  plt.plot(ns, 1/ns**4, label=r"$O(h^4)$")
  plt.grid(True)
  plt.xscale('log', basex=2)
  plt.yscale('log')
  plt.title("Convergence")
  plt.legend()
  plt.show()

#%% Question 4
n = 24
x, y_a = experiment(n)
y = ya(x)
plot(x, y, y_a)
print("L2 norm:", np.linalg.norm(y - y_a))

#%% Check Convergence
ns = np.array([2**i for i in range(6, 10)])
err = np.zeros(len(ns))

for i in range(len(ns)):
  x_, y_a_ = experiment(ns[i])
  y_ = ya(x_)
  err[i] = np.linalg.norm(y_ - y_a_)
  
#%% Plot Convergence
plotConv(ns, err)

#%% Data to save
data = np.zeros((n, 3))
data[:,0] = x
data[:,1] = y
data[:,2] = y_a

#%%
error = np.zeros((len(ns), 2))
error[:,0] = 1/ns
error[:,1] = err

#%%
DIR = 'data/4/'
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True) # Create Folder
# Write files
np.savetxt(DIR + 'data.csv', data, fmt='%.16f', delimiter=',', header='x,y,y_a', comments="")
np.savetxt(DIR + 'error.csv', error, fmt='%.16f', delimiter=',', header='n,e', comments="")

