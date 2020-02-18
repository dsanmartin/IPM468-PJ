import numpy as np
import matplotlib.pyplot as plt
#%%
class Experiment:
  
  def __init__(self, **kwargs):
    self.f = kwargs['f']    
    self.g = kwargs['g']    
    self.Sf = kwargs['Sf']
    
    
    self.Nx = kwargs['Nx']
    self.Nt = kwargs['Nt']
    
    self.L = kwargs['L']
    self.T = kwargs['T']
    
    self.h0 = kwargs['h0']
    self.u0 = kwargs['u0']
    
    self.x = np.linspace(0, self.L, self.Nx)
    self.t = np.linspace(0, self.T, self.Nt)
    
    self.dx = self.x[1] - self.x[0]
    self.dt = self.t[1] - self.t[0]



  def F(self, t, y):
    h = np.copy(y[:self.Nx])
    Q = np.copy(y[self.Nx:])
    
    #hn = np.copy(h)
    Qn = np.copy(Q)
    
    # dQ/dx
    Qn[1:-1] = (Q[2:] - Q[:-2]) / (2 * self.dx)
    
    # Inertia
    In = Q ** 2 / h + self.g * h ** 2 / 2 
    #print(In)
    
    In[1:-1] = (In[2:] - In[:-2]) / (2 * self.dx)
    
    hf = - Qn
    Qf = - (In + self.g * h * Sf(self.f, self.g, h, Q) )
    
    # Boundary
    hf[0] = hf[1]
    hf[-1] = hf[-2]
    Qf[0] = Qf[1]
    Qf[-1] = Qf[-2]
    # aa = 0
    # bb = 0
    
    # # U
    # uf[:,0] = uf[:,1] - aa * self.dx
    # uf[:,-1] = uf[:,-2] + aa * self.dx
    # uf[0,:] = uf[1,:] - bb * self.dy
    # uf[-1,:] = uf[-2,:] + bb * self.dy
    
    # # V
    # vf[:,0] = vf[:,1] - aa * self.dx
    # vf[:,-1] = vf[:,-2] + aa * self.dx
    # vf[0,:] = vf[1,:] - bb * self.dy
    # vf[-1,:] = vf[-2,:] + bb * self.dy

    return np.r_[hf.flatten(), Qf.flatten()]
    
  
  def RK4(self, F, y0):
    U = np.zeros((self.Nt, len(y0)))
    U[0] = y0
    dt = self.dt
    for k in range(self.Nt - 1):
      k1 = F(self.t[k], U[k])
      k2 = F(self.t[k] + 0.5 * dt, U[k] + 0.5 * dt * k1)
      k3 = F(self.t[k] + 0.5 * dt, U[k] + 0.5 * dt * k2)
      k4 = F(self.t[k] + dt, U[k] + dt * k3)
      U[k + 1] = U[k] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
  
    return U  
  
  
  def solvePDE(self):
    
    y0 = np.zeros(2 * self.Nx)
    y0[:self.Nx] = self.h0(self.x)
    y0[self.Nx:] = self.u0(self.x) * self.h0(self.x)
    
    y = self.RK4(self.F, y0)
    
    H = y[:, :self.Nx]#.reshape(len(self.t), self.N, self.N)
    Q = y[:, self.Nx:]#.reshape(len(self.t), self.N, self.N)
    
    return self.t, self.x, H, Q
  
#%%
x0 = 1000
L = 2000
T = 30
Nx = 50
Nt = 20000
f = 0
g = 1

h0 = lambda x: np.piecewise(x, [x < x0, x >= x0], [40, 1]) 
u0 = lambda x: x * 0
Sf = lambda f, g, h, Q: 0 * Q #f * np.abs(Q) * Q / (8 * g * h ** 3)

#%%
exp_1 = Experiment(
  f = f,
  g = g,
  L = L,
  T = T,
  Nx = Nx,
  Nt = Nt,
  h0 = h0,
  u0 = u0,
  Sf = Sf
)

#%%
t, x, H, Q = exp_1.solvePDE()
#%%
plt.plot(x, h0(x))
plt.show()
#%%
#plt.contourf(x[1:-1], t[:200], H[:200,1:-1])
plt.contourf(x, t, H)
plt.colorbar()
plt.show()
#%%
plt.contourf(x, t, Q/H)
plt.colorbar()
plt.show()