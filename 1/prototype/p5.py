import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
# Analityc solution
p = lambda x, y, z, k: (1 - np.exp(k * x * y * z * (1 - x) * (1 - y) * (1 - z))) / (1 - np.exp(k / 64))

# RHS 
f = lambda x, y, z, k: (-2*k*y*z*(1-y)*(1-z) + (k*y*z*(1-y)*(1-z)*(1-2*x))**2 - 2*k*x*z*(1-z)*(1-x) + (k*z*x*(1-z)*(1-x)*(1-2*y))**2  - 2*k*x*y*(1-x)*(1-y) + (k*x*y*(1-x)*(1-y)*(1-2*z))**2) * (-np.exp(k*x*y*z*(1-x)*(1-y)*(1-z))/(1-np.exp(k/64))) 

#%%
def plot3D(x, y, z):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(x, y, z)
  plt.colorbar(surf)
  plt.show()
  
def plotError(x, y, u, u_a):
  plt.contourf(x, y, np.abs(u - u_a))
  plt.colorbar()
  plt.plot()
  
#%%
def JacobiLoop(U0, F, h, N_iter):
  U = np.copy(U0)
  Nx, Ny, Nz = U.shape
  for n in range(N_iter):
    print(n)
    for i in range(1, Nx - 1):
      for j in range(1, Ny - 1):
        for k in range(1, Nz - 1):
          U[i,j,k] = 1/6 * (U[i+1,j,k] + U[i-1,j,k] + U[i,j+1,k] + U[i,j-1,k] + U[i,j,k+1] + U[i,j,k-1] - h ** 2 * F[i, j, k])    
  return U
          
def JacobiVec(U0, F, h, N_iter):
  U = np.copy(U0)
  for n in range(N_iter):
    U[1:-1,1:-1,1:-1] = 1/6 * (U[2:,1:-1,1:-1] + U[:-2,1:-1,1:-1] + U[1:-1,2:,1:-1] + U[1:-1,:-2,1:-1] + U[1:-1,1:-1,2:] + U[1:-1,1:-1,:-2] - h ** 2 * F[1:-1,1:-1,1:-1])
  return U

def SOR(U0, F, h, N_iter, w=0.5):
  Nx, Ny, Nz = U0.shape
  U = np.zeros((2, Nx, Ny, Nz))
  U[0] = np.copy(U0)
  n = 0
  for it in range(N_iter - 1):
    for i in range(1, Nx - 1):
      for j in range(1, Ny - 1):
        for k in range(1, Nz - 1):
          U[n+1,i,j,k] = (1 - w) * U[n,i,j,k] + w / 6 * (U[n,i+1,j,k] + U[n+1,i-1,j,k] + U[n,i,j+1,k] + U[n+1,i,j-1,k] + U[n,i,j,k+1] + U[n+1,i,j,k-1] - h ** 2 * F[i,j,k])
    U[0] = np.copy(U[n+1])
  return U[-1]
  
def experiment(f, r, Nx, Ny, Nz, N_iter, solver, w=0.5):
  x = np.linspace(0, 1, Nx + 1)
  y = np.linspace(0, 1, Ny + 1)
  z = np.linspace(0, 1, Nz + 1)
  
  X, Y = np.meshgrid(x, y)
  X3, Y3, Z3 = np.meshgrid(x, y, z)
  
  # Parameters
  h = x[1] - x[0] # h = dx = dy = dz
  F = f(X3, Y3, Z3, r)
  
  U0 = np.zeros((Nx + 1, Ny + 1, Nz + 1)) # Initial u including BC
  U = solver(U0, F, h, N_iter)
  
  return X, Y, X3, Y3, Z3, U
  
#%%
# Function parameter (k)
r = 1 
#%% Coarse mesh using Jacobi
Xc, Yc, X3c, Y3c, Z3c, Uc = experiment(f, r, 20, 20, 20, 1000, JacobiVec)

#%%
# Plot u(x,y,0.8)
plot3D(Xc, Yc, p(Xc, Yc, np.ones_like(Xc) * 0.8, r))        
plot3D(Xc, Yc, Uc[:,:,16])    
plotError(Xc, Yc, p(Xc, Yc, np.ones_like(Xc) * 0.8, r),  Uc[:,:,16])
        
# Error
print("L2 norm:", np.linalg.norm((p(X3c, Y3c, Z3c, r) - Uc).flatten()))
print("L_inf norm: ", np.linalg.norm((p(X3c, Y3c, Z3c, r) - Uc).flatten(), np.inf))
#%% Fine mesh using Jacobi
Xf, Yf, X3f, Y3f, Z3f, Uf = experiment(f, r, 50, 50, 50, 5000, JacobiVec)

#%%
# Plots      
plot3D(Xf, Yf, p(Xf, Yf, np.ones_like(Xf) * 0.8, r))        
plot3D(Xf, Yf, Uf[:,:,40])    
plotError(Xf, Yf, p(Xf, Yf, np.ones_like(Xf) * 0.8, r), Uf[:,:,40])
        
# Error
print("L2 norm:", np.linalg.norm((p(X3f, Y3f, Z3f, r) - Uf).flatten()))
print("L_inf norm: ", np.linalg.norm((p(X3f, Y3f, Z3f, r) - Uf).flatten(), np.inf))

#%% Coarse mesh using Relaxation
Xc_r, Yc_r, X3c_r, Y3c_r, Z3c_r, Uc_r = experiment(f, r, 20, 20, 20, 500, SOR, w=.5)

#%%
# Plots      
plot3D(Xc_r, Yc_r, p(Xc_r, Yc_r, np.ones_like(Xc_r) * 0.8, r))        
plot3D(Xc_r, Yc_r, Uc_r[:,:,16]) 