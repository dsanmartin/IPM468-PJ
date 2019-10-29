import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%% Functions...

# Analityc solution
p = lambda x, y, z, k: (1 - np.exp(k * x * y * z * (1 - x) * (1 - y) * (1 - z))) / (1 - np.exp(k / 64))

# RHS 
f = lambda x, y, z, k: (-2*k*y*z*(1-y)*(1-z) + (k*y*z*(1-y)*(1-z)*(1-2*x))**2 - 2*k*x*z*(1-z)*(1-x) + (k*z*x*(1-z)*(1-x)*(1-2*y))**2  - 2*k*x*y*(1-x)*(1-y) + (k*x*y*(1-x)*(1-y)*(1-2*z))**2) * (-np.exp(k*x*y*z*(1-x)*(1-y)*(1-z))/(1-np.exp(k/64))) 
  
# Jacobi implementation using loops
def Jacobi(U0, F, h, N_iter):
  U = np.copy(U0)
  Nx, Ny, Nz = U.shape
  for n in range(N_iter):
    for i in range(1, Nx - 1):
      for j in range(1, Ny - 1):
        for k in range(1, Nz - 1):
          U[i,j,k] = 1/6 * (U[i+1,j,k] + U[i-1,j,k] + U[i,j+1,k] + U[i,j-1,k] + U[i,j,k+1] + U[i,j,k-1] - h ** 2 * F[i, j, k])    
  return U
     
# Jacobi vectorized implementation using slices (only testing)   
def JacobiVec(U0, F, h, N_iter):
  U = np.copy(U0)
  for n in range(N_iter):
    U[1:-1,1:-1,1:-1] = 1/6 * (U[2:,1:-1,1:-1] + U[:-2,1:-1,1:-1] + U[1:-1,2:,1:-1] + U[1:-1,:-2,1:-1] + U[1:-1,1:-1,2:] + U[1:-1,1:-1,:-2] - h ** 2 * F[1:-1,1:-1,1:-1])
  return U

# Relaxation implementation
def SOR(U0, F, h, N_iter, w=0.5):
  Nx, Ny, Nz = U0.shape
  U = np.zeros((2, Nx, Ny, Nz)) # Just keep two steps for n and n+1
  U[0] = np.copy(U0)
  n = 0
  for it in range(N_iter):
    for i in range(1, Nx - 1):
      for j in range(1, Ny - 1):
        for k in range(1, Nz - 1):
          U[n+1,i,j,k] = (1 - w) * U[n,i,j,k] + w / 6 * (U[n,i+1,j,k] + U[n+1,i-1,j,k] + U[n,i,j+1,k] + U[n+1,i,j-1,k] + U[n,i,j,k+1] + U[n+1,i,j,k-1] - h ** 2 * F[i,j,k])
    U[n] = np.copy(U[n+1]) # Copy U^{n+1} in U^{n} for next iteration
  return U[-1]

# Experiment
def experiment(f, r, Nx, Ny, Nz, N_iter, solver, w=None):
  
  # Grid definition
  x = np.linspace(0, 1, Nx + 1)
  y = np.linspace(0, 1, Ny + 1)
  z = np.linspace(0, 1, Nz + 1)
  
  X, Y = np.meshgrid(x, y)
  X3, Y3, Z3 = np.meshgrid(x, y, z)
  
  # Parameters
  h = x[1] - x[0] # h = dx = dy = dz
  F = f(X3, Y3, Z3, r) # f evaluation for RHS
  
  U0 = np.zeros((Nx + 1, Ny + 1, Nz + 1)) # Initial u including BC
  
  if w is not None: # For relaxation
    U = solver(U0, F, h, N_iter, w)
  else: # For Jacobi
    U = solver(U0, F, h, N_iter)
  
  return X, Y, X3, Y3, Z3, U # Return matrices
  
# Plots
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
# Function parameter (k)
r = 1 

#%% Coarse mesh using Jacobi
Xc, Yc, X3c, Y3c, Z3c, Uc = experiment(f, r, 20, 20, 20, 500, Jacobi)

#%%
# Plot u(x,y,0.8)
Pc = p(Xc, Yc, np.ones_like(Xc) * 0.8, r)
plot3D(Xc, Yc, Pc)        
plot3D(Xc, Yc, Uc[:,:,16])    
plotError(Xc, Yc, Pc,  Uc[:,:,16])
        
# Error
print("L2 norm:", np.linalg.norm((p(X3c, Y3c, Z3c, r) - Uc).flatten()))
print("L_inf norm: ", np.linalg.norm((p(X3c, Y3c, Z3c, r) - Uc).flatten(), np.inf))

#%% Fine mesh using Jacobi
Xf, Yf, X3f, Y3f, Z3f, Uf = experiment(f, r, 50, 50, 50, 3000, Jacobi)

#%%
# Plots      
Pf = p(Xf, Yf, np.ones_like(Xf) * 0.8, r)
plot3D(Xf, Yf, Pf)        
plot3D(Xf, Yf, Uf[:,:,40])    
plotError(Xf, Yf, Pf, Uf[:,:,40])
        
# Error
print("L2 norm:", np.linalg.norm((p(X3f, Y3f, Z3f, r) - Uf).flatten()))
print("L_inf norm: ", np.linalg.norm((p(X3f, Y3f, Z3f, r) - Uf).flatten(), np.inf))

#%% Coarse mesh using Relaxation
Xc_r, Yc_r, X3c_r, Y3c_r, Z3c_r, Uc_r = experiment(f, r, 20, 20, 20, 200, SOR, w=1.9)

#%%
# Plots      
Pc_r = p(Xc_r, Yc_r, np.ones_like(Xc_r) * 0.8, r)
plot3D(Xc_r, Yc_r, Pc_r)        
plot3D(Xc_r, Yc_r, Uc_r[:,:,16]) 
plotError(Xc_r, Yc_r, Pc_r, Uc_r[:,:,16])
        
# Error
print("L2 norm:", np.linalg.norm((p(X3c_r, Y3c_r, Z3c_r, r) - Uc_r).flatten()))
print("L_inf norm: ", np.linalg.norm((p(X3c_r, Y3c_r, Z3c_r, r) - Uc_r).flatten(), np.inf))

#%% Fine mesh using Relaxation
Xf_r, Yf_r, X3f_r, Y3f_r, Z3f_r, Uf_r = experiment(f, r, 50, 50, 50, 500, SOR, w=1.9)

#%%
# Plots      
Pf_r = p(Xf_r, Yf_r, np.ones_like(Xf_r) * 0.8, r)
plot3D(Xf_r, Yf_r, Pf_r)        
plot3D(Xf_r, Yf_r, Uf_r[:,:,40])    
plotError(Xf_r, Yf_r, Pf_r, Uf_r[:,:,40])
        
# Error
print("L2 norm:", np.linalg.norm((p(X3f_r, Y3f_r, Z3f_r, r) - Uf).flatten()))
print("L_inf norm: ", np.linalg.norm((p(X3f_r, Y3f_r, Z3f_r, r) - Uf).flatten(), np.inf))

#%% Matrices for plots
Uc_j = np.zeros((21 * 21, 3)); Uc_j[:,0] = Xc.flatten(); Uc_j[:,1] = Yc.flatten(); Uc_j[:,2] = Uc[:,:,16].flatten()
Uf_j = np.zeros((51 * 51, 3)); Uf_j[:,0] = Xf.flatten(); Uf_j[:,1] = Yf.flatten(); Uf_j[:,2] = Uf[:,:,40].flatten()
Uc_rr = np.zeros((21 * 21, 3)); Uc_rr[:,0] = Xc.flatten(); Uc_rr[:,1] = Yc.flatten(); Uc_rr[:,2] = Uc_r[:,:,16].flatten()
Uf_rr = np.zeros((51 * 51, 3)); Uf_rr[:,0] = Xf.flatten(); Uf_rr[:,1] = Yf.flatten(); Uf_rr[:,2] = Uf_r[:,:,40].flatten()

Ec_j = np.zeros((21 * 21, 3)); Ec_j[:,0] = Xc.flatten(); Ec_j[:,1] = Yc.flatten(); Ec_j[:,2] = np.abs(Pc - Uc[:,:,16]).flatten()
Ef_j = np.zeros((51 * 51, 3)); Ef_j[:,0] = Xf.flatten(); Ef_j[:,1] = Yf.flatten(); Ef_j[:,2] = np.abs(Pf - Uf[:,:,40]).flatten()
Ec_r = np.zeros((21 * 21, 3)); Ec_r[:,0] = Xc.flatten(); Ec_r[:,1] = Yc.flatten(); Ec_r[:,2] = np.abs(Pc_r - Uc_r[:,:,16]).flatten()
Ef_r = np.zeros((51 * 51, 3)); Ef_r[:,0] = Xf.flatten(); Ef_r[:,1] = Yf.flatten(); Ef_r[:,2] = np.abs(Pf_r - Uf_r[:,:,40]).flatten()

#%% Save data
DIR = 'data/5/a/'
np.savetxt(DIR + 'Uc_j.csv', Uc_j, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'Uf_j.csv', Uf_j, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'Uc_r.csv', Uc_rr, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'Uf_r.csv', Uf_rr, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'Ec_j.csv', Ec_j, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'Ef_j.csv', Ef_j, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'Ec_r.csv', Ec_r, fmt='%.8f', delimiter=' ')
np.savetxt(DIR + 'Ef_r.csv', Ef_r, fmt='%.8f', delimiter=' ')