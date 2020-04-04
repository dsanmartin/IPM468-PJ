import numpy as np
#%%
class Experiment1D:
  
  def __init__(self, **kwargs):
    self.f = kwargs['f']    
    self.g = kwargs['g']    
    self.Sf = kwargs['Sf']
    
    self.Nx = kwargs['Nx'] + 1
    self.Nt = kwargs['Nt'] + 1
    
    self.L = kwargs['L']
    self.T = kwargs['T']
    
    self.h0 = kwargs['h0']
    self.u0 = kwargs['u0']
    
    self.x = np.linspace(0, self.L, self.Nx)
    self.t = np.linspace(0, self.T, self.Nt)
    
    self.dx = self.x[1] - self.x[0]
    self.dt = self.t[1] - self.t[0]

  
  def ev(self, q, h):
    idx = np.array((h == 0)) # indexes where h == 0
    h[idx] = 1  
    u = q / h
    l1 = u + np.sqrt(self.g * h)
    l2 = u - np.sqrt(self.g * h)
    l1[idx] = 0 
    l2[idx] = 0 
    return l1, l2
  
  
  # F components
  def F1(self, q):
    return q
  
  def F2(self, h, q):
    return q ** 2 / h + 0.5 * self.g * h ** 2
  
  # Lax-Friedrichs scheme for F components 
  def FLF1(self, hl, hr, ql, qr):
    return 0.5 * (self.F1(qr) + self.F1(ql)) - 0.5 * self.dx / self.dt * (hr - hl)
  
  def FLF2(self, hl, hr, ql, qr):
    return 0.5 * (self.F2(hr, qr) + self.F2(hl, ql)) - 0.5 * self.dx / self.dt * (qr - ql)
  
  # Rusanov scheme for F components 
  def FR1(self, hl, hr, ql, qr):
    l1l, l2l = self.ev(ql, hl)
    l1r, l2r = self.ev(qr, hr)
    c = np.maximum(np.maximum(np.abs(l1l), np.abs(l2l)), np.maximum(np.abs(l1r), np.abs(l2r)))
    return 0.5 * (self.F1(qr) + self.F1(ql)) - 0.5 * c * (hr - hl)
  
  def FR2(self, hl, hr, ql, qr):
    l1l, l2l = self.ev(ql, hl)
    l1r, l2r = self.ev(qr, hr)
    c = np.maximum(np.maximum(np.abs(l1l), np.abs(l2l)), np.maximum(np.abs(l1r), np.abs(l2r)))
    return 0.5 * (self.F2(hr, qr) + self.F2(hl, ql)) - 0.5 * c * (qr - ql)
  
  def S(self, h, q):
    return -self.g * h * self.Sf(self.f, self.g, h, q)
  
  def solvePDE(self, scheme='lf'):
    
    h = np.zeros((self.Nt, self.Nx))
    q = np.zeros((self.Nt, self.Nx))
    
    h[0] = self.h0(self.x)
    q[0] = self.u0(self.x) * self.h0(self.x)
    
    
    if scheme == 'lf': # Lax-Friedrichs scheme
    
      for n in range(self.Nt - 1):
        
        # Compute water height
        h[n+1, 1:-1] = h[n, 1:-1] \
          - self.dt / self.dx * (
            self.FLF1(h[n, 1:-1], h[n, 2:], q[n, 1:-1], q[n, 2:]) - \
            self.FLF1(h[n, :-2], h[n, 1:-1],  q[n, :-2], q[n, 1:-1])
          )
               
        # To avoid zero division, replace h and q by 1
        idx1 = np.array((h[n] == 0)) # indexes where h == 0
        h[n, idx1] = 1 
           
        # Flow
        q[n+1, 1:-1] = q[n, 1:-1] \
          - self.dt / self.dx * (
            self.FLF2(h[n, 1:-1], h[n, 2:], q[n, 1:-1], q[n, 2:]) - \
            self.FLF2(h[n, :-2], h[n, 1:-1],  q[n, :-2], q[n, 1:-1])
          )
        
        # Set values replaced by 1 with 0
        h[n, idx1] = 0
        q[n+1, idx1] = 0
        
        # Splitting S(U)
        q[n+1, 1:-1] += self.dt * self.S(h[n+1, 1:-1], q[n+1, 1:-1])
        
         # Remove S(U) correction errors
        idx2 = np.array((q[n+1] <= 0)) # indexes where q < 0
        q[n+1, idx2] = 0
        
        # Boundary
        h[n+1, 0] = h[n+1, 1]
        h[n+1,-1] = h[n+1, -2]
        q[n+1, 0] = q[n+1 ,1]
        q[n+1,-1] = q[n+1, -2]
    
    elif scheme == 'rs': # Rusanov scheme      
      for n in range(self.Nt - 1):

        # Compute water height
        h[n+1, 1:-1] = h[n, 1:-1] \
          - self.dt / self.dx * (
            self.FR1(h[n, 1:-1], h[n, 2:], q[n, 1:-1], q[n, 2:]) - \
            self.FR1(h[n, :-2], h[n, 1:-1],  q[n, :-2], q[n, 1:-1])
          )
            
        # To avoid zero division, replace h and q by 1
        idx1 = np.array((h[n] == 0)) # indexes where h = 0
        h[n, idx1] = 1
                   
        # Flow
        q[n+1, 1:-1] = q[n, 1:-1] \
          - self.dt / self.dx * (
            self.FR2(h[n, 1:-1], h[n, 2:], q[n, 1:-1], q[n, 2:]) - \
            self.FR2(h[n, :-2], h[n, 1:-1],  q[n, :-2], q[n, 1:-1])
          ) #+ self.dt * self.S(h[n, 1:-1], q[n, 1:-1])
            
        
        # Set values replaced by 1 with 0
        h[n, idx1] = 0
        q[n+1, idx1] = 0
              
        # Correction of S(U)
        q[n+1, 1:-1] += self.dt * self.S(h[n+1, 1:-1], q[n+1, 1:-1])
        
        # Remove S(U) correction errors
        idx2 = np.array((q[n+1] <= 0)) # indexes where q < 0
        q[n+1, idx2] = 0
          
        # Boundary conditions
        h[n+1, 0] = h[n+1, 1]
        h[n+1, -1] = h[n+1, -2]
        q[n+1,0] = q[n+1 ,1]
        q[n+1,-1] = q[n+1, -2]
    
    return self.t, self.x, h, q
  
class Experiment2D:
  
  def __init__(self, **kwargs):
    self.f = kwargs['f']    
    self.g = kwargs['g']    
    self.Sf = kwargs['Sf']
    
    self.Nx = kwargs['Nx'] + 1
    self.Ny = kwargs['Ny'] + 1
    self.Nt = kwargs['Nt'] + 1
    
    self.L = kwargs['L']
    self.T = kwargs['T']
    
    self.h0 = kwargs['h0']
    self.u0 = kwargs['u0']
    self.v0 = kwargs['v0']
    
    self.x = np.linspace(0, self.L, self.Nx)
    self.y = np.linspace(0, self.L, self.Ny)
    self.t = np.linspace(0, self.T, self.Nt)
    
    # GRID
    self.X, self.Y = np.meshgrid(self.x, self.y)
    
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    self.dt = self.t[1] - self.t[0]


  def evF(self, q1, h):
    idx = np.array((h <= 0)) # indexes where h == 0
    h[idx] = 1 
    u = q1 / h
    l1 = u + np.sqrt(self.g * h)
    l2 = u
    l3 = u - np.sqrt(self.g * h)
    l1[idx] = 0 
    l2[idx] = 0 
    l3[idx] = 0 
    return l1, l2, l3
  
  def evG(self, q2, h):
    idx = np.array((h <= 0)) # indexes where h == 0
    h[idx] = 1
    v = q2 / h
    l1 = v + np.sqrt(self.g * h)
    l2 = v
    l3 = v - np.sqrt(self.g * h)
    l1[idx] = 0
    l2[idx] = 0 
    l3[idx] = 0 
    return l1, l2, l3
  
  def c(self, hl, hr, ql, qr, ev):
    l1l, l2l, l3l = ev(ql, hl)
    l1r, l2r, l3r = ev(qr, hr)
    cl = np.maximum(np.maximum(np.abs(l1l), np.abs(l2l)), np.abs(l3l))
    cr = np.maximum(np.maximum(np.abs(l1r), np.abs(l2r)), np.abs(l3r))
    c = np.maximum(cl, cr)
    return c
    
  
  # F components
  def F1(self, q1):
    return q1
  
  def F2(self, h, q1):
    return q1 ** 2 / h + 0.5 * self.g * h ** 2
  
  def F3(self, h, q1, q2):
    return q1 * q2 / h
  
  # G components
  def G1(self, q2):
    return q2
  
  def G2(self, h, q1, q2):
    return q1 * q2 / h
  
  def G3(self, h, q2):
    return q2 ** 2 / h + 0.5 * self.g * h ** 2
  
  # Lax-Friedrichs scheme for F components 
  def FLF1(self, hl, hr, q1l, q1r):
    return 0.5 * (self.F1(q1r) + self.F1(q1l)) - 0.5 * self.dx / self.dt * (hr - hl)
  
  def FLF2(self, hl, hr, q1l, q1r):
    return 0.5 * (self.F2(hr, q1r) + self.F2(hl, q1l)) - 0.5 * self.dx / self.dt * (q1r - q1l)
  
  def FLF3(self, hl, hr, q1l, q1r, q2l, q2r):
    return 0.5 * (self.F3(hr, q1r, q2r) + self.F3(hl, q1l, q2l)) - 0.5 * self.dx / self.dt * (q2r - q2l)
  
  # Lax-Friedrichs scheme for G components 
  def GLF1(self, hl, hr, q2l, q2r):
    return 0.5 * (self.G1(q2r) + self.G1(q2l)) - 0.5 * self.dx / self.dt * (hr - hl)
  
  def GLF2(self, hl, hr, q1l, q1r, q2l, q2r):
    return 0.5 * (self.G2(hr, q1r, q2r) + self.G2(hl, q1l, q2l)) - 0.5 * self.dx / self.dt * (q1r - q1l)
  
  def GLF3(self, hl, hr, q2l, q2r):
    return 0.5 * (self.G3(hr, q2r) + self.G3(hl, q2l)) - 0.5 * self.dx / self.dt * (q2r - q2l)
  
  # Rusanov scheme for F components 
  def FR1(self, hl, hr, q1l, q1r):
    c = self.c(hl, hr, q1l, q1r, self.evF)
    return 0.5 * (self.F1(q1r) + self.F1(q1l)) - 0.5 * c * (hr - hl)
  
  def FR2(self, hl, hr, q1l, q1r):
    c = self.c(hl, hr, q1l, q1r, self.evF)
    return 0.5 * (self.F2(hr, q1r) + self.F2(hl, q1l)) - 0.5 * c * (q1r - q1l)
  
  def FR3(self, hl, hr, q1l, q1r, q2l, q2r):
    c = self.c(hl, hr, q1l, q1r, self.evF)
    return 0.5 * (self.F3(hr, q1r, q2r) + self.F3(hl, q1l, q2l)) - 0.5 * c * (q2r - q2l)
  
  # Rusanov scheme for G components 
  def GR1(self, hl, hr, q2l, q2r):
    c = self.c(hl, hr, q2l, q2r, self.evG)
    return 0.5 * (self.G1(q2r) + self.G1(q2l)) - 0.5 * c * (hr - hl)
  
  def GR2(self, hl, hr, q1l, q1r, q2l, q2r):
    c = self.c(hl, hr, q2l, q2r, self.evG)
    return 0.5 * (self.G2(hr, q1r, q2r) + self.G2(hl, q1l, q2l)) - 0.5 * c * (q1r - q1l)
  
  def GR3(self, hl, hr, q2l, q2r):
    c = self.c(hl, hr, q2l, q2r, self.evG)
    return 0.5 * (self.G3(hr, q2r) + self.G3(hl, q2l)) - 0.5 * c * (q2r - q2l)
  
  # S components
  def S2(self, h, q1):
    return - self.g * h * self.Sf(self.f, self.g, h, q1)
  
  def S3(self, h, q2):
    return - self.g * h * self.Sf(self.f, self.g, h, q2)

  
  def solvePDE(self, scheme='lf'):
        
    h = np.zeros((self.Nt, self.Nx, self.Ny))
    q1 = np.zeros((self.Nt, self.Nx, self.Ny))
    q2 = np.zeros((self.Nt, self.Nx, self.Ny))
    
    h[0] = self.h0(self.X, self.Y)
    q1[0] = self.u0(self.X, self.Y) * self.h0(self.X, self.Y)
    q2[0] = self.v0(self.X, self.Y) * self.h0(self.X, self.Y)
    
    if scheme == 'lf': # Lax-Friedrichs scheme
    
      for n in range(self.Nt - 1):
        
        h[n+1, 1:-1, 1:-1] = h[n, 1:-1, 1:-1] - \
            self.dt / self.dx * (
            self.FLF1(h[n, 1:-1, 1:-1], h[n, 1:-1, 2:], q1[n, 1:-1, 1:-1], q1[n, 1:-1, 2:]) - \
            self.FLF1(h[n, 1:-1, :-2], h[n, 1:-1, 1:-1],  q1[n, 1:-1, :-2], q1[n, 1:-1, 1:-1])
          ) - \
            self.dt / self.dy * (
            self.GLF1(h[n, 1:-1, 1:-1], h[n, 2:, 1:-1], q2[n, 1:-1, 1:-1], q2[n, 2:, 1:-1]) - \
            self.GLF1(h[n, :-2, 1:-1], h[n, 1:-1, 1:-1], q2[n, :-2, 1:-1], q2[n, 1:-1, 1:-1])
          )
          
        # # To avoid zero division, replace h and q by 1
        idx1 = np.array((h[n] <= 1e-2)) # indexes where h == 0
        #idx2 = np.array((q1[n] == 0)) # indexes where q == 0
        #idx3 = np.array((q2[n] == 0)) # indexes where q == 0
        h[n, idx1] = 1 
        #q1[n, idx2] = 1 
        #q2[n, idx3] = 1 
        
        q1[n+1, 1:-1, 1:-1] = q1[n, 1:-1, 1:-1] \
          - self.dt / self.dx * (
            self.FLF2(h[n, 1:-1, 1:-1], h[n, 1:-1, 2:], q1[n, 1:-1, 1:-1], q1[n, 1:-1, 2:]) - \
            self.FLF2(h[n, 1:-1, :-2], h[n, 1:-1, 1:-1],  q1[n, 1:-1, :-2], q1[n, 1:-1, 1:-1])
          ) \
          - self.dt / self.dy * (
            self.GLF2(h[n, 1:-1, 1:-1], h[n, 2:, 1:-1], q1[n, 1:-1, 1:-1], q1[n, 2:, 1:-1], q2[n, 1:-1, 1:-1], q2[n, 2:, 1:-1]) - \
            self.GLF2(h[n, :-2, 1:-1], h[n, 1:-1, 1:-1], q1[n, :-2, 1:-1], q1[n, 1:-1, 1:-1], q2[n, :-2, 1:-1], q2[n, 1:-1, 1:-1])
          )
          
        # Splitting of S(U)
        q1[n+1, 1:-1, 1:-1] += self.dt * self.S2(h[n, 1:-1, 1:-1], q1[n, 1:-1, 1:-1])
        
        q2[n+1, 1:-1, 1:-1] = q2[n, 1:-1, 1:-1] \
          - self.dt / self.dx * (
            self.FLF3(h[n, 1:-1, 1:-1], h[n, 1:-1, 2:], q1[n, 1:-1, 1:-1], q1[n, 1:-1, 2:], q2[n, 1:-1, 1:-1], q2[n, 1:-1, 2:]) - \
            self.FLF3(h[n, 1:-1, :-2], h[n, 1:-1, 1:-1], q1[n, 1:-1, :-2], q1[n, 1:-1, 1:-1], q2[n, 1:-1, :-2], q2[n, 1:-1, 1:-1])
          ) \
          - self.dt / self.dy * (
            self.GLF3(h[n, 1:-1, 1:-1], h[n, 2:, 1:-1], q2[n, 1:-1, 1:-1], q2[n, 2:, 1:-1]) - \
            self.GLF3(h[n, :-2, 1:-1], h[n, 1:-1, 1:-1], q2[n, :-2, 1:-1], q2[n, 1:-1, 1:-1])
          ) #+ self.dt * self.S3(h[n, 1:-1, 1:-1], q2[n, 1:-1, 1:-1])
        
        # Splitting S(U)
        q2[n+1, 1:-1, 1:-1] += self.dt * self.S3(h[n, 1:-1, 1:-1], q2[n, 1:-1, 1:-1])
        
        
        
        
        # Remove S(U) correction errors
        # idx = np.array((h[n+1] < 1e-2) | (h[n+1] > 4e1)) # indexes where q < 0
        # idx2 = np.array((q1[n+1] < 1e-2) | (h[n+1] > 4e1)) # indexes where q < 0
        # idx3 = np.array((q2[n+1] < 1e-2) | (h[n+1] > 4e1)) # indexes where q < 0
        # h[n+1, idx] = 0
        # h[n+1, idx2] = 0
        # h[n+1, idx3] = 0
        # q1[n+1, idx2] = 0
        # q2[n+1, idx3] = 0
        
        # Set values replaced by 1 with 0
        h[n, idx1] = 0
        q1[n+1, idx1] = 0
        q2[n+1, idx1] = 0
        
        # Boundary
        h[n+1, :, 0] = h[n+1, :, 1]
        h[n+1, :,-1] = h[n+1, :,-2]
        q1[n+1, :, 0] = q1[n+1, :, 1]
        q1[n+1, :,-1] = q1[n+1, :,-2]
        
        h[n+1, 0, :] = h[n+1, 1, :]
        h[n+1,-1, :] = h[n+1,-2, :]
        q2[n+1, 0, :] = q2[n+1, 1, :]
        q2[n+1,-1, :] = q2[n+1,-2, :]
        
    elif scheme == 'rs':
      for n in range(self.Nt - 1):
        
        h[n+1, 1:-1, 1:-1] = h[n, 1:-1, 1:-1] \
          - self.dt / self.dx * (
            self.FR1(h[n, 1:-1, 1:-1], h[n, 1:-1, 2:], q1[n, 1:-1, 1:-1], q1[n, 1:-1, 2:]) - \
            self.FR1(h[n, 1:-1, :-2], h[n, 1:-1, 1:-1],  q1[n, 1:-1, :-2], q1[n, 1:-1, 1:-1])
          ) \
          - self.dt / self.dy * (
            self.GR1(h[n, 1:-1, 1:-1], h[n, 2:, 1:-1], q2[n, 1:-1, 1:-1], q2[n, 2:, 1:-1]) - \
            self.GR1(h[n, :-2, 1:-1], h[n, 1:-1, 1:-1], q2[n, :-2, 1:-1], q2[n, 1:-1, 1:-1])
          )
          
        # # To avoid zero division, replace h and q by 1
        idx1 = np.array((h[n] == 0)) # indexes where h == 0
        #idx2 = np.array((q1[n] == 0)) # indexes where q == 0
        #idx3 = np.array((q2[n] == 0)) # indexes where q == 0
        h[n, idx1] = 1 
        #q1[n, idx2] = 1 
        #q2[n, idx3] = 1 
        
        q1[n+1, 1:-1, 1:-1] = q1[n, 1:-1, 1:-1] \
          - self.dt / self.dx * (
            self.FR2(h[n, 1:-1, 1:-1], h[n, 1:-1, 2:], q1[n, 1:-1, 1:-1], q1[n, 1:-1, 2:]) - \
            self.FR2(h[n, 1:-1, :-2], h[n, 1:-1, 1:-1],  q1[n, 1:-1, :-2], q1[n, 1:-1, 1:-1])
          ) \
          - self.dt / self.dy * (
            self.GR2(h[n, 1:-1, 1:-1], h[n, 2:, 1:-1], q1[n, 1:-1, 1:-1], q1[n, 2:, 1:-1], q2[n, 1:-1, 1:-1], q2[n, 2:, 1:-1]) - \
            self.GR2(h[n, :-2, 1:-1], h[n, 1:-1, 1:-1], q1[n, :-2, 1:-1], q1[n, 1:-1, 1:-1], q2[n, :-2, 1:-1], q2[n, 1:-1, 1:-1])
          ) 
            
        # Splitting of S(U)
        q1[n+1, 1:-1, 1:-1] += self.dt * self.S2(h[n, 1:-1, 1:-1], q1[n, 1:-1, 1:-1])
          
        q2[n+1, 1:-1, 1:-1] = q2[n, 1:-1, 1:-1] \
          - self.dt / self.dx * (
            self.FR3(h[n, 1:-1, 1:-1], h[n, 1:-1, 2:], q1[n, 1:-1, 1:-1], q1[n, 1:-1, 2:], q2[n, 1:-1, 1:-1], q2[n, 1:-1, 2:]) - \
            self.FR3(h[n, 1:-1, :-2], h[n, 1:-1, 1:-1], q1[n, 1:-1, :-2], q1[n, 1:-1, 1:-1], q2[n, 1:-1, :-2], q2[n, 1:-1, 1:-1])
          ) \
          - self.dt / self.dy * (
            self.GR3(h[n, 1:-1, 1:-1], h[n, 2:, 1:-1], q2[n, 1:-1, 1:-1], q2[n, 2:, 1:-1]) - \
            self.GR3(h[n, :-2, 1:-1], h[n, 1:-1, 1:-1], q2[n, :-2, 1:-1], q2[n, 1:-1, 1:-1])
          ) 
            
        # Splitting S(U)
        q2[n+1, 1:-1, 1:-1] += self.dt * self.S3(h[n, 1:-1, 1:-1], q2[n, 1:-1, 1:-1])
        
        # Set values replaced by 1 with 0
        h[n, idx1] = 0
        q1[n+1, idx1] = 0
        q2[n+1, idx1] = 0
            
        # Remove S(U) correction errors
        # idx2 = np.array((q1[n+1] < 0)) # indexes where q < 0
        # idx3 = np.array((q2[n+1] < 0)) # indexes where q < 0
        # q1[n+1, idx2] = 0
        # q2[n+1, idx3] = 0
        
        # Boundary
        h[n+1, :, 0] = h[n+1, :, 1]
        h[n+1, :,-1] = h[n+1, :,-2]
        q1[n+1, :, 0] = q1[n+1, :, 1]
        q1[n+1, :,-1] = q1[n+1, :,-2]
        
        h[n+1, 0, :] = h[n+1, 1, :]
        h[n+1,-1, :] = h[n+1,-2, :]
        q2[n+1, 0, :] = q2[n+1, 1, :]
        q2[n+1,-1, :] = q2[n+1,-2, :]
    
    return self.t, self.X, self.Y, h, q1, q2
  
