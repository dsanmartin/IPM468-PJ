import numpy as np
import matplotlib.pyplot as plt
  
#%%
f = lambda x: x ** 3
ya = lambda x: (6 * np.cos(1) - 3) * np.cos(x) / np.sin(1) + 6 * np.sin(x) + x ** 3 - 6 * x
#dy = lambda x: (3 - 6 * np.cos(1)) * np.sin(x) / np.sin(1) + 6 * np.cos(x) + 3 * x ** 2 - 6

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
  
  print(x1)
  print(x2)
  


def createMatricesBK(n, h):
  # A matrix
  A = np.zeros((n, n))
  np.fill_diagonal(A[1:], 1) # lower diagonal
  A += A.T # upper diagonal
  np.fill_diagonal(A, 10) # main diagonal
  A[0,:2] = np.array([1, 3.08510638]) 
  A[-1,-2:] = np.array([-3.08510638, -1]) 

  # B matrix
  B = np.zeros((n, n))
  np.fill_diagonal(B[1:], -6)
  B += B.T
  np.fill_diagonal(B, 12)
  #B[0,:2] = np.array([6, -6])
  #B[-1,-2:] = np.array([-6, 6])
  B[0,:4] = np.array([-1.14893617, -1.27659574,  2.42553191,  h*0.12765957])
  B[-1,-4:] = -np.array([-1.14893617, -1.27659574,  2.42553191,  h*0.12765957])
  
  B *= -2 / h ** 2 
  #print(A); print(B)

  return A, B


def createMatrices(n, h):
  # A matrix
  A = np.zeros((n, n))
  np.fill_diagonal(A[1:], 10) # lower diagonal
  A += A.T # upper diagonal
  np.fill_diagonal(A, 1) # main diagonal
  #A[0,:2] = np.array([11, -2]) 
  #A[-1,-2:] = np.array([-2, 11]) 
  A[0,:2] = np.array([1, -2.63636364]) 
  A[-1,-2:] = np.array([-2.63636364, 1]) 

  # B matrix
  B = np.zeros((n, n))
  np.fill_diagonal(B[1:], 12)
  B += B.T
  np.fill_diagonal(B, -24)
  
  #B[0,:2] = np.array([-6, 6])
  #B[-1,-2:] = np.array([6, -6])
  B[0,:4] = np.array([-3., 5.45454545, -2.45454545, 0*-0.54545455*h])
  B[-1,-4:] = np.array([-3., 5.45454545, -2.45454545, 0*0.54545455*h])
   
  B /= h ** 2 
  
  #print(A); print(B)

  return A, B
  
  
def experiment(n):
  h = 1/(n)
  i = np.arange(1, n+1)
  
  x = (i - .5) * h
  
  A, B = createMatrices(n, h)
  F = f(x)
  I = np.eye(n)
  
  y = np.linalg.solve(np.dot(np.linalg.inv(A), B) + I, F)
  print(x)
  
  return x, y

  
def plot(x, y, y_a):
  plt.plot(x, y, label="Real")
  plt.plot(x, y_a, label="Approximation")
  plt.grid(True)
  plt.legend()
  plt.show()

#%%
#solveSystem()  
#print(c_1)
#%%
n = 24
x, y_a = experiment(n)

y = ya(x)

plot(x, y, y_a)

print(np.linalg.norm(y - y_a))

#%% Data to save
data = np.zeros((n, 3))
data[:,0] = x
data[:,1] = y
data[:,2] = y_a

#%%
DIR = 'data/4/'
np.savetxt(DIR + 'data.csv', data, fmt='%.8f', delimiter=',', header='x,y,y_a', comments="")


