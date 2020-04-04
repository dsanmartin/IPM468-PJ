import numpy as np

class Experiment:
  
  def __init__(self, **kwargs):
    self.H = kwargs['H']
    self.f = kwargs['f']
    self.g = kwargs['g']
    self.b = kwargs['b']
    
    self.Nt = kwargs['Nt'] + 1
    self.Nx = kwargs['Nx'] + 1
    self.Ny = kwargs['Ny'] + 1
    
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
    
    
    self.bc = kwargs['bc']


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
    if self.bc == 1:
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
      
    elif self.bc == 2:
      uf[:,0] = (4 * uf[:,1] - uf[:,2]) / 3
      uf[:,-1] = (4 * uf[:,-2] - uf[:,-3]) / 3 
      uf[0,:] = (4 * uf[1,:] - uf[2,:]) / 3
      uf[-1,:] = (4 * uf[-2,:] - uf[-3,:]) / 3 
      
      vf[:,0] = (4 * vf[:,1] - vf[:,2]) / 3
      vf[:,-1] = (4 * vf[:,-2] - vf[:,-3]) / 3 
      vf[0,:] = (4 * vf[1,:] - vf[2,:]) / 3
      vf[-1,:] = (4 * vf[-2,:] - vf[-3,:]) / 3 
      
      hf[:,0] = (4 * hf[:,1] - hf[:,2]) / 3
      hf[:,-1] = (4 * hf[:,-2] - hf[:,-3]) / 3 
      hf[0,:] = (4 * hf[1,:] - hf[2,:]) / 3
      hf[-1,:] = (4 * hf[-2,:] - hf[-3,:]) / 3 
      
      
    
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
  
  
  
  ### Finite volume ###
  def FF(self, U):
    h, u, v = U
    return np.array([self.H * u, self.g * h, v * 0])
  
  def GG(self, U):
    h, u, v = U
    return np.array([self.H * v, u * 0, self.g * h])
  
  
  # Lax-Friedrichs
  def LFF(self, Ul, Ur):
    c = self.dx / self.dt
    return 0.5 * (self.FF(Ul) + self.FF(Ur)) - 0.5 * c * (Ur - Ul)
  
  def LFG(self, Ub, Uu):
    c = self.dy / self.dt
    return 0.5 * (self.GG(Ub) + self.GG(Uu)) - 0.5 * c * (Uu - Ub)
  
  #Rusanov
  def ev(self):
    l1 = np.sqrt(self.H * self.g)
    l2 = 0
    l3 = -np.sqrt(self.H * self.g)
    return l1, l2, l3

  # 
  def RSF(self, Ul, Ur):
    c = np.max(self.ev())
    return 0.5 * (self.FF(Ul) + self.FF(Ur)) - 0.5 * c * (Ur - Ul)

  def RSG(self, Ub, Uu):
    c = np.max(self.ev())
    return 0.5 * (self.GG(Ub) + self.GG(Uu)) - 0.5 * c * (Uu - Ub)
  
  # Fluxes computation  
  def Fx(self, U):
    h, u, v = U
    Ul = np.array([h[1:-1, :-2], u[1:-1, :-2], v[1:-1, :-2]]) 
    Uc = np.array([h[1:-1, 1:-1], u[1:-1, 1:-1], v[1:-1, 1:-1]])
    Ur = np.array([h[1:-1, 2:], u[1:-1, 2:], v[1:-1, 2:]]) 
    
    #return (self.LFF(Uc, Ur) - self.LFF(Ul, Uc)) / self.dx
    return (self.RSF(Uc, Ur) - self.RSF(Ul, Uc)) / self.dx
  
  def Gy(self, U):
    h, u, v = U
    Ub = np.array([h[:-2, 1:-1], u[:-2, 1:-1], v[:-2, 1:-1]])
    Uc = np.array([h[1:-1, 1:-1], u[1:-1, 1:-1], v[1:-1, 1:-1]])
    Uu = np.array([h[2:, 1:-1], u[2:, 1:-1], v[2:, 1:-1]])
    
    #return (self.LFG(Uc, Uu) - self.LFG(Ub, Uc)) / self.dy
    return (self.RSG(Uc, Uu) - self.RSG(Ub, Uc)) / self.dy
    
  def S(self, U):
    h, u, v = U
    return np.array([h * 0, self.f * v - self.b * u, -self.f * u - self.b * v])
  
  def BC(self, U):
      # h
      U[0, :, 0] = U[0, :, 1]; U[0, :, -1] = U[0, :, -2]
      U[0, 0, :] = U[0, 1, :]; U[0, -1, :] = U[0, -2, :]
      
      # u
      U[1, :, 0] = U[1, :, 1]; U[1, :, -1] = U[1, :, -2]
      U[1, 0, :] = U[1, 1, :]; U[1, -1, :] = U[1, -2, :]
      
      # v
      U[2, :, 0] = U[2, :, 1]; U[2, :, -1] = U[2, :, -2]
      U[2, 0, :] = U[2, 1, :]; U[2, -1, :] = U[2, -2, :]

      
  def FFF(self, t, U):
    return -(self.Fx(U) + self.Gy(U))
  
  def euler(self, U0):
    U = np.zeros((self.Nt, 3, self.Ny, self.Nx))
    U[0] = U0
    
    for n in range(self.Nt - 1):
      U[n+1, :, 1:-1, 1:-1] = U[n, :, 1:-1, 1:-1] + self.dt * self.FFF(self.t[n], U[n, :]) # Conservative
      
      U[n+1, :] += self.dt * self.S(U[n+1, :]) # Splitting
      
      self.BC(U[n+1]) # Boundary conditions
      
    return U
  
  def RK4VF(self, U0):
    U = np.zeros((self.Nt, 3, self.Ny, self.Nx))
    U[0] = U0
    dt = self.dt
    k1 = np.zeros((3, self.Ny, self.Nx))
    k2 = np.zeros_like(k1)
    k3 = np.zeros_like(k1)
    k4 = np.zeros_like(k1)
    
    for n in range(self.Nt - 1):
      k1[:, 1:-1, 1:-1] = self.FFF(self.t[n], U[n, :])
      k2[:, 1:-1, 1:-1] = self.FFF(self.t[n] + 0.5 * dt, U[n, :] + 0.5 * dt * k1)
      k3[:, 1:-1, 1:-1] = self.FFF(self.t[n] + 0.5 * dt, U[n, :] + 0.5 * dt * k2)
      k4[:, 1:-1, 1:-1] = self.FFF(self.t[n] + dt, U[n, :] + dt * k3)
      U[n+1, :, 1:-1, 1:-1] = U[n, :, 1:-1, 1:-1] + \
        (1/6) * dt * (k1[:, 1:-1, 1:-1] + 2 * k2[:, 1:-1, 1:-1] + 2 * k3[:, 1:-1, 1:-1] + k4[:, 1:-1, 1:-1])
      
      U[n+1, :] += self.dt * self.S(U[n+1, :]) # Splitting
      
      self.BC(U[n+1]) # Boundary conditions
  
    return U 
    
  

  def solveVF(self, method='euler'):
    # Domain grid
    X, Y = np.meshgrid(self.x, self.y)
    
    # Initial conditions    
    U0 = np.array([
      self.h0(X, Y),
      self.u0(X, Y),
      self.v0(X, Y)
    ])
    
    if method == 'euler':
      solver = self.euler
    elif method == 'rk4':
      solver = self.RK4VF
    
    # Solve
    U = solver(U0)
      
    return self.t, X, Y, U[:, 0], U[:, 1], U[:, 2]
      
      
    
    
    
    
    
    
    
    
    