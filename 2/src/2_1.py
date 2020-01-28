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
    
    #self.K = kwargs['K']
    
    self.dx = self.L / self.N
    self.dy = self.dx
    
    self.x = np.arange(0, self.L, self.dx)#np.linspace(0, self.L, self.N)
    self.y = np.copy(self.x)
    
    #self.dx = self.dy = self.x[1] - self.x[0]
    
    self.dt = self.dx / 100
    self.t = np.arange(0, self.T_max, self.dt)
    
    
    #self.dt = self.T_max / self.K
    #self.t = np.linspace(0, self.T_max, self.K)
    
    
    print(len(self.t))


  def F(self, t, y):
    h = np.copy(y[:self.N ** 2].reshape((self.N, self.N)))
    u = np.copy(y[self.N ** 2: 2 * self.N ** 2].reshape((self.N, self.N)))
    v = np.copy(y[2 * self.N ** 2:].reshape((self.N, self.N)))
    
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
# Initial conditions
h0 = lambda x, y, R, hp: 1 + hp * (np.sqrt((x - .5)**2 + (y - .5)**2) <= R) # Initial 
u0 = lambda x, y: x * 0
v0 = lambda x, y: x * 0
h0g = lambda x, y: 1 + 0.11 * np.exp(-100*((x-.5)**2 + (y-.5)**2))
#%% 
# x = np.linspace(0, 1, 100)
# y = np.copy(x)
# X, Y = np.meshgrid(x, y)
# H0 = h0(X, Y, 0.1, 0.1)
# HG = h0g(X, Y)
# #%%

# plt.imshow(H0, extent=[0, 1, 0, 1])
# plt.colorbar()
# plt.plot()
# #%%
# plt.imshow(HG, extent=[0, 1, 0, 1])
# plt.colorbar()
# plt.plot()

#%%

H_ = 10
f_ = 0
g_ = 1
b_ = 2

T_max_ = .5

L_ = 1
N_ = 100

#K_ = 500

exp_1 = Experiment(
  H = H_,
  f = f_,
  g = g_,
  b = b_,
  L = L_,
  N = N_,
  T_max = T_max_,
  h0 = lambda x, y: h0(x, y, .1, .1),
  #h0 = h0g,
  u0 = u0,
  v0 = v0,
  #K = K_
)

#%%
t, X, Y, H, U, V = exp_1.solvePDE()
#%%
for k in range(len(t)):
  if k % 100 == 0:
    print(k)
    #plt.contourf(X, Y, H[k], vmin=np.min(H), vmax=np.max(H))#H[-1])
    plt.imshow(H[k], #vmin=np.min(H), vmax=np.max(H), 
               origin="lower", extent=[0, 1, 0, 1])
    #plt.quiver(X, Y, U[10], V[0])
    plt.colorbar()
    plt.show()

#%%
from mpl_toolkits import mplot3d

fig = plt.figure()

k = -1
ax = plt.gca(projection='3d')
sf = ax.plot_surface(X, Y, H[k])#, vmin=np.min(H), vmax=np.max(H))#H[-1])
#plt.quiver(X, Y, U[10], V[0])
plt.show()




