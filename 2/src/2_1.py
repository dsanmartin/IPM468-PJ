import numpy as np
import matplotlib.pyplot as plt
#%%
class Experiment:
  
  def __init__(self, **kwargs):
    self.H = kwargs['H']
    self.f = kwargs['f']
    self.g = kwargs['g']
    self.b = kwargs['b']
    
    self.L = kwargs['L']
    self.N = kwargs['N']
    
    self.T_max = kwargs['T_max']
    
    self.h0 = kwargs['h0']
    self.u0 = kwargs['u0']
    self.v0 = kwargs['v0']
    
    self.dx = self.L / self.N
    self.dy = self.dx
    
    self.x = np.arange(0, self.L, self.dx)#np.linspace(0, self.L, self.N)
    self.y = np.copy(self.x)
    
    #self.dx = self.dy = self.x[1] - self.x[0]
    
    self.dt = self.dx / 100
    self.t = np.arange(0, self.T_max, self.dt)
    
    print(len(self.t))
    
    
#    print("x: ", self.x, "dx: ", self.dx)
#    print("y: ", self.y, "dy: ", self.dy)
#    print("t: ", self.t, "dt: ", self.dt)

  def F(self, t, y):
    h = np.copy(y[:self.N ** 2].reshape((self.N, self.N)))
    u = np.copy(y[self.N ** 2: 2 * self.N ** 2].reshape((self.N, self.N)))
    v = np.copy(y[2 * self.N ** 2:].reshape((self.N, self.N)))
    
    hx = np.copy(h)
    hy = np.copy(h)
    ux = np.copy(u)
    vy = np.copy(v)
    
#    hx[1:-1, 1:-1] = (h[2:,1:-1] - 2 * h[1:-1,1:-1] + h[:-2,1:-1]) / self.dx / self.dx
#    hy[1:-1, 1:-1] = (h[1:-1,2:] - 2 * h[1:-1,1:-1] + h[1:-1,:-2]) / self.dy / self.dy
#    
#    ux[1:-1, 1:-1] = (u[2:,1:-1] - 2 * u[1:-1,1:-1] + u[:-2,1:-1]) / self.dx / self.dx
#    vy[1:-1, 1:-1] = (v[1:-1,2:] - 2 * v[1:-1,1:-1] + v[1:-1,:-2]) / self.dy / self.dy
    
#    hx[1:-1, 1:-1] = (h[1:-1,2:] - 2 * h[1:-1,1:-1] + h[1:-1,:-2]) / self.dx / self.dx
#    hy[1:-1, 1:-1] = (h[2:,1:-1] - 2 * h[1:-1,1:-1] + h[:-2,1:-1]) / self.dy / self.dy
#    
#    vy[1:-1, 1:-1] = (v[2:,1:-1] - 2 * v[1:-1,1:-1] + v[:-2,1:-1]) / self.dy / self.dy
#    ux[1:-1, 1:-1] = (u[1:-1,2:] - 2 * u[1:-1,1:-1] + u[1:-1,:-2]) / self.dx / self.dx
    
    hx[1:-1, 1:-1] = (h[1:-1,2:] - h[1:-1,:-2]) / (2 * self.dx)
    hy[1:-1, 1:-1] = (h[2:,1:-1] - h[:-2,1:-1]) / (2 * self.dy)
    
    vy[1:-1, 1:-1] = (v[2:,1:-1] - v[:-2,1:-1]) / (2 * self.dy)
    ux[1:-1, 1:-1] = (u[1:-1,2:] - u[1:-1,:-2]) / (2 * self.dx)
    
    
    hf = - self.H * (ux + vy)
    uf = self.f * v - self.g * hx - self.b * u
    vf = -self.f * u - self.g * hy - self.b * v
    
    # Boundary
    aa = 1
    bb = 1
    
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
#    hf[:,0] = np.zeros(self.N)
#    hf[:,-1] = np.zeros(self.N)
#    hf[0,:] = np.zeros(self.N)
#    hf[-1,:] = np.zeros(self.N)
    
#    print(hf.shape)
#    
#    X, Y = np.meshgrid(self.x, self.y)
#    plt.contourf(X, Y, hf)
#    plt.colorbar()
#    plt.show()
#    
    return np.r_[hf.flatten(), uf.flatten(), vf.flatten()]
    
  
  def RK4(self, F, y0):
    L = len(self.t)
    U = np.zeros((L, len(y0)))
    U[0] = y0
    #dt = self.t[1] - self.t[0]
    dt = self.dt
    for k in range(L-1):
      k1 = F(self.t[k], U[k])
      k2 = F(self.t[k] + 0.5 * dt, U[k] + 0.5 * dt * k1)
      k3 = F(self.t[k] + 0.5 * dt, U[k] + 0.5 * dt * k2)
      k4 = F(self.t[k] + dt, U[k] + dt * k3)
      U[k + 1] = U[k] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
  
    return U  
  
  
  def solvePDE(self):
    X, Y = np.meshgrid(self.x, self.y)
    
    H0 = self.h0(X, Y)
    U0 = self.u0(X, Y)
    V0 = self.v0(X, Y)
    
    y0 = np.zeros(3 * self.N ** 2)
    y0[:self.N ** 2] = H0.flatten()
    y0[self.N ** 2: 2 * self.N ** 2] = U0.flatten()
    y0[2 * self.N ** 2:] = V0.flatten()
    
    y = self.RK4(self.F, y0)
    
    H = y[:, :self.N ** 2].reshape(len(self.t), self.N, self.N)
    U = y[:, self.N ** 2: 2 * self.N ** 2].reshape(len(self.t), self.N, self.N)
    V = y[:, 2 * self.N ** 2:].reshape(len(self.t), self.N, self.N)
    
    return self.t, X, Y, H, U, V
  
  
#%%
#def h0(x, y):
#  return np.ones_like(x) #x * 0 + 1.0
h0 = lambda x, y: x * 0 + 1 
u0 = lambda x, y: x * 0
v0 = lambda x, y: x * 0

def h0_2(x, y, R, hp):
  center = (int(len(x)/2), int(len(y)/2))
  Y, X = np.ogrid[:len(y), :len(x)]
  dist_from_center = np.sqrt((X - center[0]) **2 + (Y-center[1]) **2)
  mask = dist_from_center <= R
  return h0(x,y) + hp * mask
  

H_ = .05
f_ = 0
g_ = 1
b_ = 2

T_max_ = .15

L_ = 1
N_ = 100

exp_1 = Experiment(
  H = H_,
  f = f_,
  g = g_,
  b = b_,
  L = L_,
  N = N_,
  T_max = T_max_,
  h0 = lambda x, y: h0_2(x, y, .1, .1),
  u0 = u0,
  v0 = v0
)
#%%
t, X, Y, H, U, V = exp_1.solvePDE()
#%%
for k in range(len(t)):
  if k % 100 == 0:
    print(k)
    plt.contourf(X, Y, H[k], vmin=np.min(H), vmax=np.max(H))#H[-1])
    #plt.quiver(X, Y, U[10], V[0])
    plt.colorbar()
    plt.show()

#%%
from mpl_toolkits import mplot3d

fig = plt.figure()

k = 2000
ax = plt.gca(projection='3d')
print(k)
sf = ax.plot_surface(X, Y, H[-1])#, vmin=np.min(H), vmax=np.max(H))#H[-1])
#plt.quiver(X, Y, U[10], V[0])
fig.colorbar(sf)
plt.show()
