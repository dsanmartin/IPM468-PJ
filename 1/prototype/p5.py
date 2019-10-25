import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
def JacobiLoop(U, F, h, N_iter):
  Nx, Ny, Nz = U.shape
  for n in range(N_iter):
    print(n)
    for i in range(1, Nx - 1):
      for j in range(1, Ny - 1):
        for k in range(1, Nz - 1):
          U[i,j,k] = 1/6 * (U[i+1,j,k] + U[i-1,j,k] + U[i,j+1,k] + U[i,j-1,k] + U[i,j,k+1] + U[i,j,k-1] - h ** 2 * F[i, j, k]) 
          
def JacobiVec(U, F, h, N_iter):
  for n in range(N_iter):
    U[1:-1,1:-1,1:-1] = 1/6 * (U[2:,1:-1,1:-1] + U[:-2,1:-1,1:-1] + U[1:-1,2:,1:-1] + U[1:-1,:-2,1:-1] + U[1:-1,1:-1,2:] + U[1:-1,1:-1,:-2] - h ** 2 * F[1:-1,1:-1,1:-1])

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
  return U[-1]
  
#%%
# Analityc solution
p = lambda x, y, z, k: (1 - np.exp(k * x * y * z * (1 - x) * (1 - y) * (1 - z))) / (1 - np.exp(k / 64))

# RHS 
f = lambda x, y, z, k: (-2*k*y*z*(1-y)*(1-z) + (k*y*z*(1-y)*(1-z)*(1-2*x))**2 - 2*k*x*z*(1-z)*(1-x) + (k*z*x*(1-z)*(1-x)*(1-2*y))**2  - 2*k*x*y*(1-x)*(1-y) + (k*x*y*(1-x)*(1-y)*(1-2*z))**2) * (-np.exp(k*x*y*z*(1-x)*(1-y)*(1-z))/(1-np.exp(k/64))) 

# Function parameter (k)
r = 1 
#%% COARSE MESH
Nx_c, Ny_c, Nz_c = 20, 20, 20
x_c = np.linspace(0, 1, Nx_c + 1)
y_c = np.linspace(0, 1, Ny_c + 1)
z_c = np.linspace(0, 1, Nz_c + 1)

X_c, Y_c = np.meshgrid(x_c, y_c)
X3_c, Y3_c, Z3_c = np.meshgrid(x_c, y_c, z_c)

# Parameters
h_c = x_c[1] - x_c[0] # h = dx = dy = dz
F_c = f(X3_c, Y3_c, Z3_c, r)

#%% Solve 
N_iter = 1000
U_c = np.zeros((Nx_c + 1, Ny_c + 1, Nz_c + 1)) # Initial u including BC
JacobiVec(U_c, F_c, h_c, N_iter)

#%%
# Plot u(x,y,0.8)
plot3D(X_c, Y_c, p(X_c, Y_c, np.ones_like(X_c) * 0.8, r))        
plot3D(X_c, Y_c, U_c[:,:,16])    

#%% 
plotError(X_c, Y_c, p(X_c, Y_c, np.ones_like(X_c) * 0.8, r),  U_c[:,:,16])
        
# Error
print("L2 norm:", np.linalg.norm((p(x_c, y_c, z_c, r) - U_c).flatten()))
print("L_inf norm: ", np.linalg.norm((p(x_c, y_c, z_c, r) - U_c).flatten(), np.inf))

#%% FINE MESH
Nx_f, Ny_f, Nz_f = 50, 50, 50
x_f = np.linspace(0, 1, Nx_f + 1)
y_f = np.linspace(0, 1, Ny_f + 1)
z_f = np.linspace(0, 1, Nz_f + 1)

X_f, Y_f = np.meshgrid(x_f, y_f)
X3_f, Y3_f, Z3_f = np.meshgrid(x_f, y_f, z_f)

h_f = x_f[1] - x_f[0] # h = dx = dy = dz
F_f = f(X3_f, Y3_f, Z3_f, r) # THS evaluation

#%% Solve fine mesh
N_iter = 5000
U_f = np.zeros((Nx_f + 1, Ny_f + 1, Nz_f + 1))  # Initial u including BC
JacobiVec(U_f, F_f, h_f, N_iter)

#%%
# Plots      
plot3D(X_f, Y_f, p(X_f, Y_f, np.ones_like(X_f) * 0.8, r))        
plot3D(X_f, Y_f, U_f[:,:,40])    
#%%
plotError(X_f, Y_f, p(X_f, Y_f, np.ones_like(X_f) * 0.8, r),  U_f[:,:,40])
        
# Error
print("L2 norm:", np.linalg.norm((p(x_f, y_f, z_f, r) - U_f).flatten()))
print("L_inf norm: ", np.linalg.norm((p(x_f, y_f, z_f, r) - U_f).flatten(), np.inf))


#%%
N_iter = 5000
U0_c = np.zeros((Nx_c + 1, Ny_c + 1, Nz_c + 1))
U_s = SOR(U0_c, F_c, h_c, N_iter, w=1.1)
#%%
# Plots      
plot3D(X_c, Y_c, p(X_c, Y_c, np.ones_like(X_c) * 0.8, r))        
plot3D(X_c, Y_c, U_s[:,:,16]) 