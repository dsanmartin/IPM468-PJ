import pathlib
import numpy as np
from dambreak import Experiment1D
from plot import plot1D, plot2D, plot3D, compare

#%%
def h_(x, t, h0, x0, g, c0, xA, xB):

  # Data to return
  o = np.zeros_like(x)
  # Cases 
  # First case
  idx_1 = np.array((x <= xA(t))) # Index where condition 1 is 
  o[idx_1] = h0 
  # Second case
  idx_2 = np.array((x > xA(t)) & (x <= xB(t)))
  o[idx_2] = (-(x[idx_2]-x0)/t[idx_2] + 2 * c0) ** 2 / (9 * g)
  # Third case, just keep zeros
  
  return o

  
def u_(x, t, x0, c0, xA, xB):
  o = np.zeros_like(x)
  # Cases, first and third case just keep zeros
  # Second case
  idx = np.array(((x >= xA(t)) & (x <= xB(t))))
  idx2 = np.array((t == 0)) # indexes where t == 0
  t[idx2] = 1
  o[idx] = 2 / 3 * ((x[idx] - x0) / t[idx] + c0)
  o[idx2] = 0
  
  return o

xA_ = lambda t, c0, x0: x0 - c0 * t
xB_ = lambda t, c0, x0: x0 + 2 * c0 * t
  
#%% Analytic
# Parameters
h_0 = 40
g = 1
x_0 = 1000
c_0 = np.sqrt(g * h_0)

xA = lambda t: xA_(t, c_0, x_0)
xB = lambda t: xB_(t, c_0, x_0)

h = lambda x, t: h_(x, t, h_0, x_0, g, c_0, xA, xB)
u = lambda x, t: u_(x, t, x_0, c_0, xA, xB)

x_i, x_f = 0, 2000
t_i, t_f = 0, 40
Nx = 201
Nt = 201
x = np.linspace(x_i, x_f, Nx)
t = np.linspace(t_i, t_f, Nt)
X, T = np.meshgrid(x, t)

#%% Dressler
HD = h(X, T)
UD = u(X, T)

#%% Initial condition
plot1D(x, HD[0])

#%% Evolution
plot2D(x, t, HD)

#%% 3D
plot3D(X, T, HD)

#%% Initial condition
plot1D(x, UD[0])

#%% Evolution
plot2D(x, t, UD)

#%% 3D
plot3D(X, T, UD)

#%% Numerical Dressler
h_0 = 40
h_d = 1e-14
x_0 = 1000
L = 2000
T = 40 # 1
Nx = 200 #200
Nt = 10 # 5000
g = 1
f = .1 #8 * g / (40 ** 2)

h0_ = lambda x, x_0, h_0, h_d: np.piecewise(x, [x <= x_0, x > x_0], [h_0, h_d]) 
h0 = lambda x: h0_(x, x_0, h_0, h_d) # Initial condition h
u0 = lambda x: x * 0  # Initial condition u
Sf = lambda f, g, h, Q: f * np.abs(Q) * Q / (8 * g * h ** 3) # Friction function
#%%
dressler = Experiment1D(
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
tl, xl, Hl, Ql = dressler.solvePDE('lf')

#%% Rusanov
tr, xr, Hr, Qr = dressler.solvePDE('rs')

#%%
k = -1
plot1D(xl, Hl[k])
#%%
for k in range(Nt):
  print(k)
  plot1D(xl, Hl[k])

#%%
for k in range(300, Nt):
  print(k)
  plot1D(xr, Hr[k])

#%%
plot2D(xl, tl, Hl)
plot2D(xl, tl, Ql/Hl)

#%%
plot2D(xr, tr, Hr)
plot2D(xr, tr, Qr/Hr)

#%% Lax-Friedichs scheme numerical evolution
XLF, TLF = np.meshgrid(xl, tl[::10])
plot3D(XLF, TLF, Hl[::10,:])

#%%
XRS, TRS = np.meshgrid(xr, tr[::10])
plot3D(XRS, TRS, Hr[::10,:])


#%%
print("h(x,t)")
compare(xl, HD[-1], Hl[-1], Hr[-1])
print("Error Lax-Friedrichs: ", np.linalg.norm(HD[-1] - Hl[-1]))
print("Error Rusanov: ", np.linalg.norm(HD[-1] - Hr[-1]))

print("u(x,t)")
compare(xl, UD[-1], Ql[-1] / Hl[-1], Qr[-1] / Hr[-1])
print("Error Lax-Friedrichs: ", np.linalg.norm(UD[-1] - Ql[-1] / Hl[-1]))
print("Error Rusanov: ", np.linalg.norm(UD[-1] - Qr[-1] / Hr[-1]))

#%%Save data
DIR = 'data/3/3/' # Directory name
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True) # Create Folder
#%% Generate data

#Evolution of solution
M, N = 51, 51
data1 = np.zeros((M * N, 8)) # Evolution

Ul = Ql / Hl
Ur = Qr / Hr

data1[:,0] = XLF[::1,::4].flatten()
data1[:,1] = TLF[::1,::4].flatten()
data1[:,2] = HD[::4,::4].flatten()
data1[:,3] = Hl[::10,::4].flatten()
data1[:,4] = Hr[::10,::4].flatten()
data1[:,5] = UD[::4,::4].flatten()
data1[:,6] = Ul[::10,::4].flatten()
data1[:,7] = Ur[::10,::4].flatten()

np.savetxt(DIR + 'dressler_1D.csv', data1, fmt='%.16f', delimiter=' ', header='x t a l r u ul ur', comments="") # Save data

#%%
data2 = np.zeros((201, 7)) # Last value

data2[:,0] = xr
data2[:,1] = HD[-1]
data2[:,2] = Hl[-1]
data2[:,3] = Hr[-1]
data2[:,4] = UD[-1]
data2[:,5] = Ul[-1]
data2[:,6] = Ur[-1]
np.savetxt(DIR + 'compare_dressler_1D.csv', data2, fmt='%.16f', delimiter=' ', header='x h l r', comments="") # Save data
