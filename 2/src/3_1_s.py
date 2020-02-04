import numpy as np
import matplotlib.pyplot as plt
#%%
class Experiment:
  
  def __init__(self, **kwargs):
    self.g = kwargs['g']
    self.h = kwargs['h']
    
    
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
    
    hn = np.copy(h)
    Qn = np.copy(Q)
    
    hn[1:-1] = (h[2:] - h[:-2]) / (2 * self.dx)
    Qn[1:-1] = (Q[2:] - h[:-2]) / (2 * self.dx)

    
    hf = - Qn
    Qf = self.f * v - self.g * hx - self.b * u
    
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
  

x0 = 1000
L = 2000
Nx = 100

h0 = lambda x: np.piecewise(x, [x < x0, x >= x0], [40, 0]) 

def h0_(x):
  o = np.zeros_like(x)
  idx = np.array((x < x0))
  o[idx] = 40
  return o

x = np.linspace(0, L, Nx)




#%%
plt.plot(x, h0(x))
plt.plot(x, h0_(x))
plt.show()