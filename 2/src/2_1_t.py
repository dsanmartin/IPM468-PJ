import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#%%
class Experiment:
  
  def __init__(self, **kwargs):
    self.H = kwargs['H']
    self.f = kwargs['f']
    self.g = kwargs['g']
    self.b = kwargs['b']
    
    self.Nt = kwargs['Nt']
    self.Nx = kwargs['Nx']
    self.Ny = kwargs['Ny']
    
    self.xf = kwargs['xf']
    self.yf = kwargs['yf']
    self.tf = kwargs['tf']
    
    self.h0 = kwargs['h0']
    self.u0 = kwargs['u0']
    self.v0 = kwargs['v0']
    
    
    self.x = np.linspace(0, self.xf, self.Nx)
    self.y = np.linspace(0, self.yf, self.Ny)
    self.t = np.linspace(0, self.tf, self.Nt)
    
    #self.dx = self.dy = self.x[1] - self.x[0]
    
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    self.dt = self.t[1] - self.t[0]
    


  def F(self, t, y):
    h = np.copy(y[:self.Nx * self.Ny].reshape((self.Nx, self.Ny)))
    u = np.copy(y[self.Nx * self.Ny: 2 * self.Nx * self.Ny].reshape((self.Nx, self.Ny)))
    v = np.copy(y[2 * self.Nx * self.Ny:].reshape((self.Nx, self.Ny)))
    
    hx = np.copy(h)
    hy = np.copy(h)
    ux = np.copy(u)
    vy = np.copy(v)
    
    hx[1:-1, 1:-1] = (h[1:-1,2:] - h[1:-1,:-2]) / (2 * self.dx)
    hy[1:-1, 1:-1] = (h[2:,1:-1] - h[:-2,1:-1]) / (2 * self.dy)
    
    vy[1:-1, 1:-1] = (v[2:,1:-1] - v[:-2,1:-1]) / (2 * self.dy)
    ux[1:-1, 1:-1] = (u[1:-1,2:] - u[1:-1,:-2]) / (2 * self.dx)
    
    
    hf = - self.H * (ux + vy)
    uf = self.f * v - self.g * hx - self.b * u
    vf = -self.f * u - self.g * hy - self.b * v
    
    # Boundary
    aa = 0
    bb = 0
    
    # U
    uf[:,0] = uf[:,1] - aa * self.dx
    uf[:,-1] = uf[:,-2] + aa * self.dx
    uf[0,:] = uf[1,:] - bb * self.dy
    uf[-1,:] = uf[-2,:] + bb * self.dy
    
    # V
    vf[:,0] = vf[:,1] - aa * self.dx
    vf[:,-1] = vf[:,-2] + aa * self.dx
    vf[0,:] = vf[1,:] - bb * self.dy
    vf[-1,:] = vf[-2,:] + bb * self.dy
    
    # H
    hf[:,0] = hf[:,1] - aa * self.dx
    hf[:,-1] = hf[:,-2] + aa * self.dx
    hf[0,:] = hf[1,:] - bb * self.dy
    hf[-1,:] = hf[-2,:] + bb * self.dy
    
    return np.r_[hf.flatten(), uf.flatten(), vf.flatten()]
    
  
  def RK4(self, F, y0):
    L = len(self.t)
    U = np.zeros((L, len(y0)))
    U[0] = y0
    dt = self.dt
    for k in range(L-1):
      k1 = F(self.t[k], U[k])
      k2 = F(self.t[k] + 0.5 * dt, U[k] + 0.5 * dt * k1)
      k3 = F(self.t[k] + 0.5 * dt, U[k] + 0.5 * dt * k2)
      k4 = F(self.t[k] + dt, U[k] + dt * k3)
      U[k + 1] = U[k] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
  
    return U  
  
  
  def solvePDE(self):
    
    # Domain grid
    X, Y = np.meshgrid(self.x, self.y)
    
    # Initial conditions
    H0 = self.h0(X, Y)
    U0 = self.u0(X, Y)
    V0 = self.v0(X, Y)
    
    # Vectorize initial conditions
    y0 = np.zeros(3 * self.Nx * self.Ny)
    y0[:self.Nx * self.Ny] = H0.flatten()
    y0[self.Nx * self.Ny: 2 * self.Nx * self.Ny] = U0.flatten()
    y0[2 * self.Nx * self.Ny:] = V0.flatten()
    
    # Solve with RK4
    y = self.RK4(self.F, y0)
    
    # Reshape approximations
    H = y[:, :self.Nx * self.Ny].reshape(len(self.t), self.Nx, self.Ny)
    U = y[:, self.Nx * self.Ny: 2 * self.Nx * self.Ny].reshape(len(self.t), self.Nx, self.Ny)
    V = y[:, 2 * self.Nx * self.Ny:].reshape(len(self.t), self.Nx, self.Ny)
    
    # Return domain and approximations
    return self.t, X, Y, H, U, V
#%%
# Plots
def plot1D(x, y):
  plt.plot(x, y)
  plt.grid(True)
  plt.show()   
  
def plot2D(z):
  #plt.contourf(X, Y, H[k], vmin=np.min(H), vmax=np.max(H)))
  plt.imshow(z, vmin=np.min(z), vmax=np.max(z), 
             origin="lower", extent=[0, 1, 0, 1])
  #plt.quiver(X, Y, U[10], V[0])
  plt.colorbar()
  plt.show()
  
def plot3D(x, y, z):
  ax = plt.gca(projection='3d')
  ax.plot_surface(x, y, z)
  plt.show()
  
#%%
def airy(x, t, H, k, w):
  return .5 * H * np.cos(k * x - w * t)
#%%
# Initial conditions
h0 = lambda x, y, R, hp: 1 + hp * (np.sqrt((x - .5)**2 + (y - .5)**2) <= R) # Initial 
u0 = lambda x, y: x * 0
v0 = lambda x, y: x * 0
#h0g = lambda x, y: 1 + 0.11 * np.exp(-10*((x-.5)**2 + (y-.5)**2))

#%%

# Model parameters
H_ = .1
f_ = 0
g_ = 1
b_ = 2

# Domain limits
xf = 1
yf = 1
tf = 2

# Domain grid size
Nt = 500
Nx = 100
Ny = Nx

# Create experiment
exp_1 = Experiment(
  H = H_,
  f = f_,
  g = g_,
  b = b_,
  Nx = Nx,
  Ny = Ny,
  Nt = Nt,
  xf = xf,
  yf = yf,
  tf = tf,
  h0 = lambda x, y: h0(x, y, .1, .1),
  #h0 = h0g,
  u0 = u0,
  v0 = v0,
)

#%%
exp_1.H = .1
t1, X1, Y1, H1, U1, V1 = exp_1.solvePDE()

#%%
plot1D(t1, H1[:, Ny//2, Nx//2])

#%%
for k in range(len(t1)):
  if k % 100 == 0: 
    #plot2D(H1[k])
    plt.imshow(H1[k], interpolation='gaussian')
    plt.show()
    
#%%
plot3D(X1, Y1, H1)

#%%
N = 500
x = np.linspace(0, xf, N)
t = np.linspace(0, tf, N)
T = 0.05
k = 2 * np.pi / xf
w = 2 * np.pi / T
eta = airy(x, t, .1, k, w)
plot1D(x, eta)
plot1D(t, eta)
# #%%
# XX, TT = np.meshgrid(x,t)
# Z = airy(XX, TT, 1, k, w)
# #%%
# plt.contourf(XX, TT, Z)
# plt.show()

