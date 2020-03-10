import numpy as np
from embalse import Experiment2D
from plot import *
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

def h0_(x, y, x0, y0):
  H = np.zeros((len(x), len(y))) 
  for i in range(len(x)):
    for j in range(len(y)):
      #if x[i,j] <= x0 and y[i,j] <= y0:
      if x[i] <= x0 and y[j] <= y0:
        H[i, j] = 40
  return H
  
#%%
# 
# x_i, x_f = 0, 2000
# y_i, y_f = 0, 2000
# t_i, t_f = 0, 5
# Nx = 500
# Ny = 500
# Nt = 100

# x = np.linspace(x_i, x_f, Nx)
# y = np.linspace(y_i, y_f, Ny)
# t = np.linspace(t_i, t_f, Nt)
# X, Y = np.meshgrid(x, y)

x0 = 1000
y0 = 1000
L = 2000
T = 10 #40
Nx = 100
Ny = 100
Nt = 2000
f = 1#0
g = 1#1


dx = L / (Nx - 1)
dy = L / (Ny - 1)
dt = L / Nt
print(dx, dy, dt)
print(dt/dx, dt/dy)


#h0 = lambda x, y: h0_(x, y, x0, y0)
h0 = lambda x, y, R, hp: 1 + hp * (np.sqrt((x - 1000)**2 + (y - 1000)**2) <= R) # Initial 
u0 = lambda x, y: x * 0
v0 = lambda x, y: y * 0
Sf = lambda f, g, h, Q: f * np.abs(Q) * Q / (8 * g * h ** 3)
#%%
#plt.imshow(h0(X, Y), origin="lower")
#plot2D(X, Y, h0(x, y))

x = np.linspace(0, L, Nx)
X, Y = np.meshgrid(x, x)
j = lambda x, y: h0(x, y, 200, 40)
plot2D(X, Y, j(X, Y))
#%%
exp_1 = Experiment2D(
  f = f,
  g = g,
  L = L,
  T = T,
  Nx = Nx,
  Ny = Ny,
  Nt = Nt,
  h0 = j,
  u0 = u0,
  v0 = v0,
  Sf = Sf
)

#%%
t, x, y, H, Q1, Q2 = exp_1.solvePDE('lf')
#%%
#plot2D(x, t, H[1])
n = 2
#plt.imshow(H[n], origin="lower")
plot2D(x, y, H[n])
#plt.colorbar()

#%% Initial condition
plot1D(x, h_(x, T[0]))

#%% Evolution
plot2D(x, t, h_(X, T))

#%%
n = 1
X, Y = np.meshgrid(x, y)
#plot3D(X, Y, H[500])
plot3D(X, Y, H[n])
#plot3D(X, T, h_(X, T))