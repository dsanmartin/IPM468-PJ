import numpy as np
from plot import *
from embalse import Experiment1D
#%%
h0 = 40
g = 1
x0 = 0#1000
c0 = np.sqrt(g * h0)

xA = lambda t: -c0 * t + x0
xB = lambda t: 2 * c0 * t + x0
  

def h_(x, t):
  # Data to return
  o = np.zeros_like(x)
  # Cases 
  # First case
  idx_1 = np.array((x <= xA(t))) # Index where condition 1 is 
  o[idx_1] = h0 
  # Second case
  idx_2 = np.array((x >= xA(t)) & (x <= xB(t)))
  o[idx_2] = (-x[idx_2]/t[idx_2] + 2 * c0) ** 2 / (9 * g)
  # Third case, just keep zeros
  
  return o

  
def u_(x, t):
  o = np.zeros_like(x)
  # Cases, first and third case just keep zeros
  # Second case
  idx = np.array(((x >= xA(t)) & (x <= xB(t))))
  o[idx] = 2 / 3 * (x[idx] / t[idx] + c0)
  
  return o
  
#%% Analitic
x_i, x_f = -5, 5 #0, 2000
t_i, t_f = 0, 5
Nx = 500
Nt = 100
x = np.linspace(x_i, x_f, Nx)
t = np.linspace(t_i, t_f, Nt)
X, T = np.meshgrid(x, t)

#%% Initial condition
plot1D(x, h_(x, T[-90]))

#%% Evolution
plot2D(x, t, h_(X, T))

#%%
plot3D(X, T, h_(X, T))

#%% Numerical
h_0 = 40
h_d = 10 * 0
x0 = 1000
L = 2000
T = 40 # 1
Nx = 200
Nt = 2000 # 5000
f = 1#5 #0
g = 1#9.8 #1

h0 = lambda x: np.piecewise(x, [x < x0, x >= x0], [h_0, h_d]) 
u0 = lambda x: x * 0
Sf = lambda f, g, h, Q: f * np.abs(Q) * Q / (8 * g * h ** 3)
#%%
exp_1 = Experiment1D(
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
tl, xl, Hl, Ql = exp_1.solvePDE('lf')

#%% Rusanov
tr, xr, Hr, Qr = exp_1.solvePDE('rs')

#%%
k = 1500
#plot1D(xl, Hl[k])
plot1D(xr, Hr[k])

#%%
plot2D(x, t, H)