import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
# Functions to use
p = lambda x, y: np.cos(x + y) * np.sin(x - y)
f = lambda x, y: -4 * p(x, y)

# Conjugate gradient implementation
def ConjugateGradient(A, b, x, conv=1e-6):
  r = b - np.dot(A, x)
  p = r
  i = 0
  while np.linalg.norm(r) > conv:# and i <= max_it:
    a = np.dot(p.T, r) / (np.dot(p.T, np.dot(A, r)))
    x = x + a * p  
    r = r - a * np.dot(A, p)
    b = - np.dot(p.T, np.dot(A, r)) / np.dot(np.dot(A, p).T, p)
    p = r + b * p
    i += 1
    
  return x

# Create A and b 
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

# Experiment
def experiment(h):
  Nx, Ny = int(1/h), int(1/h)
  
  # Grid definition
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
  #u = np.linalg.solve(A, b)
  
  # Numerical solution
  U = np.copy(P) # Boundary
  U[1:-1,1:-1] = u.reshape((Nx-1), (Ny-1)) # Approximation of interior points
  
  return X, Y, U, P

# 2D plot
def plot2D(x, y, u):
  plt.contourf(x, y, u)
  plt.colorbar()
  plt.show()
  
#%% Experiments
X1, Y1, U1, P1 = experiment(0.1)  
X2, Y2, U2, P2 = experiment(0.05)  
X3, Y3, U3, P3 = experiment(0.025)  
#%% Plots
plot2D(X1, Y1, U1)
plot2D(X1, Y1, P1)
plot2D(X1, Y1, np.abs(U1-P1))

#%%
plot2D(X2, Y2, U2)
plot2D(X2, Y2, P2)
plot2D(X2, Y2, np.abs(U2 - P2))

#%%
plot2D(X3, Y3, U3)
plot2D(X3, Y3, P3)
plot2D(X3, Y3, np.abs(U3-P3))

#%% Error
e1 = np.linalg.norm((U1-P1).flatten(), np.inf)
e2 = np.linalg.norm((U2-P2).flatten(), np.inf)
e3 = np.linalg.norm((U3-P3).flatten(), np.inf)

print(e1, e2, e3)

#%% Structures to save
M1, N1 = U1.shape
M2, N2 = U2.shape
M3, N3 = U3.shape

U1f = np.zeros((M1 * N1, 3)); U1f[:,0] = X1.flatten(); U1f[:,1] = Y1.flatten(); U1f[:,2] = U1.flatten()
U2f = np.zeros((M2 * N2, 3)); U2f[:,0] = X2.flatten(); U2f[:,1] = Y2.flatten(); U2f[:,2] = U2.flatten()
U3f = np.zeros((M3 * N3, 3)); U3f[:,0] = X3.flatten(); U3f[:,1] = Y3.flatten(); U3f[:,2] = U3.flatten()
E1f = np.zeros((M1 * N1, 3)); E1f[:,0] = X1.flatten(); E1f[:,1] = Y1.flatten(); E1f[:,2] = np.abs(U1-P1).flatten()
E2f = np.zeros((M2 * N2, 3)); E2f[:,0] = X2.flatten(); E2f[:,1] = Y2.flatten(); E2f[:,2] = np.abs(U2-P2).flatten()
E3f = np.zeros((M3 * N3, 3)); E3f[:,0] = X3.flatten(); E3f[:,1] = Y3.flatten(); E3f[:,2] = np.abs(U3-P3).flatten()

#%% Save data
DIR = 'data/5/b/'
np.savetxt(DIR + 'U1.csv', U1f, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'U2.csv', U2f, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'U3.csv', U3f, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'E1.csv', E1f, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'E2.csv', E2f, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'E3.csv', E3f, fmt='%.8f', delimiter=' ')

