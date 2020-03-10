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
    #Qn[1:-1] = (Q[2:] - Q[:-2]) / (2 * self.dx)
    Qn[1:-1] = 0.5 * ((Q[2:] + Q[:-2]) / self.dt - (Q[2:] - Q[:-2]) / self.dx  ) 
    
    # Inertia
    In = Q ** 2 / h + self.g * h ** 2 / 2 
    #print(In)
    
    #In[1:-1] = (In[2:] - In[:-2]) / (2 * self.dx)
    In[1:-1] = 0.5 * ((In[2:] + In[:-2]) / self.dt - (In[2:] - In[:-2]) / self.dx  ) 
    
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
  
  def ev(self, q, h):
    l1 = q / h + np.sqrt(self.g * h)
    l2 = q / h - np.sqrt(self.g * h)
    return l1, l2
  
  def solvePDE(self, scheme='lf'):
    
    h = np.zeros((self.Nt, self.Nx))
    q = np.zeros((self.Nt, self.Nx))
    
    h[0] = self.h0(self.x)
    q[0] = self.u0(self.x) * self.h0(self.x)
    
    
    if scheme == 'lf': # Lax-Friedichs scheme
    
      for n in range(self.Nt - 1):
        h[n+1, 1:-1] = 0.5 * (h[n, 2:] + h[n, :-2]) - 0.5 * self.dt / self.dx * (q[n, 2:] - q[n, :-2])
        q[n+1, 1:-1] = 0.5 * (q[n, 2:] + q[n, :-2]) - \
          0.5 * self.dt / self.dx * (q[n, 2:] ** 2 / h[n, 2:] + self.g *  h[n, 2:] ** 2 / 2 - q[n, :-2] ** 2 / h[n, :-2] + self.g *  h[n, :-2] ** 2 / 2) \
          - self.dt * self.g * h[n, 1:-1] * self.Sf(self.f, self.g, h[n, 1:-1], q[n, 1:-1])
          
        # Boundary
        h[n+1, 0] = h[n+1, 1]
        h[n+1, -1] = h[n+1, -2]
        q[n+1,0] = q[n+1 ,1]
        q[n+1,-1] = q[n+1, -2]
    
    elif scheme == 'rs': # Rusanov scheme      
      for n in range(self.Nt - 1):
        l1_l, l2_l = self.ev(q[n, :-2], h[n, :-2])
        l1_r, l2_r = self.ev(q[n, 2:], h[n, 2:])
        s1 = np.maximum(np.abs(l1_l), np.abs(l2_l))
        s2 = np.maximum(np.abs(l1_r), np.abs(l2_r))
        c = np.maximum(s1, s2)
        h[n+1, 1:-1] = 0.5 * c * (h[n, 2:] + h[n, :-2]) - 0.5 * (q[n, 2:] - q[n, :-2])
        q[n+1, 1:-1] = 0.5 * c * (q[n, 2:] + q[n, :-2]) - \
          0.5 * (q[n, 2:] ** 2 / h[n, 2:] + self.g *  h[n, 2:] ** 2 / 2 - q[n, :-2] ** 2 / h[n, :-2] + self.g *  h[n, :-2] ** 2 / 2) \
          - self.dt * self.g * h[n, 1:-1] * Sf(self.f, self.g, h[n, 1:-1], q[n, 1:-1])
          
        # Boundary
        h[n+1, 0] = h[n+1, 1]
        h[n+1, -1] = h[n+1, -2]
        q[n+1,0] = q[n+1 ,1]
        q[n+1,-1] = q[n+1, -2]
    
    return self.t, self.x, h, q
  
#%%
x0 = 1000
L = 2000
T = 10
Nx = 100
Nt = 5000
f = 5#0
g = 9.8#1

h0 = lambda x: np.piecewise(x, [x < x0, x >= x0], [40, 1]) 
u0 = lambda x: x * 0
Sf = lambda f, g, h, Q: f * np.abs(Q) * Q / (8 * g * h ** 3)


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
t, x, H, Q = exp_1.solvePDE('lf')
#%%
plt.plot(x, h0(x))
plt.show()
#%%
#plt.contourf(x[1:-1], t[:200], H[:200,1:-1])
plt.contourf(x, t[::10], H[::10])
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar()
plt.show()
#%%
plt.contourf(x, t, Q/H)
plt.colorbar()
plt.show()
#%%
plt.plot(x, H[-1])
plt.show()