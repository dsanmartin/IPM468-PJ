import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
# Functions to use
p = lambda x, y: np.cos(x + y) * np.sin(x - y)
f = lambda x, y: -4 * p(x, y)

def plot2D(x, y, u):
  plt.contourf(x, y, u)
  plt.colorbar()
  plt.show()

def ConjugateGradient(A, b, x, conv=1e-6, max_it=2500):
  r = b - np.dot(A, x)
  p = r
  i = 0
  while np.linalg.norm(r) > conv and i <= max_it:
    a = np.dot(p.T, r) / (np.dot(p.T, np.dot(A, r)))
    x = x + a * p  
    r = r - a * np.dot(A, p)
    b = - np.dot(p.T, np.dot(A, r)) / np.dot(np.dot(A, p).T, p)
    p = r + b * p
    i += 1
    
  return x

def createSystem(Nx, Ny, h, F, P):
  # Create A matrix
  A = np.zeros(((Nx-1)*(Ny-1), (Nx-1)*(Ny-1)))
  I = np.eye(Nx-1, Ny-1) # Identity matrix
  D = np.zeros((Nx-1, Ny-1)) # For 5 points stencil matrix
  np.fill_diagonal(D[1:], 1) # Lower diagonal
  D += D.T # Upper diagonal
  np.fill_diagonal(D, -4) # Main diagonal
  # Fill A matrix
  # Upper block
  A[:Nx-1,:Ny-1] = D
  A[:Nx-1,Ny-1:2*(Ny-1)] = I
  # Lower block
  A[-(Nx-1):,-(Ny-1):] = D
  A[-(Nx-1):,-2*(Ny-1):-(Ny-1)] = I
  # Inner blocks
  for i in range(1, Nx-2):
    A[i*(Nx-1):(i+1)*(Nx-1),i*(Ny-1):(i+1)*(Ny-1)] = D
    A[i*(Nx-1):(i+1)*(Nx-1),(i-1)*(Ny-1):i*(Ny-1)] = I
    A[i*(Nx-1):(i+1)*(Nx-1),(i+1)*(Ny-1):(i+2)*(Ny-1)] = I
    
  # Create b vector
  B = h**2 * F[1:-1,1:-1] # Inside 
  B[0] += P[0,1:-1] # South boundary
  B[-1] += P[-1,1:-1] # North boundary
  B[:,0] += P[1:-1,0] # West boundary
  B[:,-1] += P[1:-1,-1] # East boundary
  
  b = B.flatten('F') # Flatten column-major order (vectorization)

  return A, b

def experiment(h):
  Nx, Ny = int(1/h), int(1/h)
  
  x = np.linspace(0, 1, Nx + 1)
  y = np.linspace(0, 1, Ny + 1)
  
  # Grid to evaluate
  X, Y = np.meshgrid(x, y)
  
  F = f(X, Y) # RHS evaluation
  P = p(X, Y) # For boundary conditions
  
  # Build A and b
  A, b = createSystem(Nx, Ny, h, F, P)
  x0 = np.zeros(len(b)) # Initial guess 
  
  # Solve
  u = ConjugateGradient(A, b, x0)
  
  # Numerical solution
  U = np.copy(P) # Boundary
  U[1:-1,1:-1] = u.reshape((Nx-1), (Ny-1)) # Approximation of interior points
  
  return X, Y, U, P
  
#%% Experiments

X1, Y1, U1, P1 = experiment(0.1)  
X2, Y2, U2, P2 = experiment(0.05)  
X3, Y3, U3, P3 = experiment(0.025)  
#%%
plot2D(X1, Y1, U1)
plot2D(X1, Y1, P1)

#%%
plot2D(X2, Y2, U2)
plot2D(X2, Y2, P2)

#%%
plot2D(X3, Y3, U3)
plot2D(X3, Y3, P3)