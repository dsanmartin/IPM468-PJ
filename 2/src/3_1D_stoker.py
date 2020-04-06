import pathlib
import numpy as np
from dambreak import Experiment1D
from plot import plot1D, plot2D, plot3D, compare

#%%
def h_(x, t, h0, hm, hd, x0, g, c0, xA, xB, xC):
  
  # Data to return
  o = np.zeros_like(x)
  # Cases 
  # First case
  idx_1 = np.array((x < xA(t))) # Index where condition 1 is 
  o[idx_1] = h0
  # Second case
  idx_2 = np.array((x >= xA(t)) & (x <= xB(t)))
  idx2 = np.array((t == 0)) # indexes where t == 0
  t[idx2] = 1
  o[idx_2] = (-(x[idx_2]-x0)/t[idx_2] + 2 * c0) ** 2 / (9 * g)
  # Third case
  idx_3 = np.array((x > xB(t)) & (x <= xC(t)))
  o[idx_3] = hm
  # Fourth case
  idx_4 = np.array((x > xC(t)))
  o[idx_4] = hd
  
  return o

  
def u_(x, t, x0, c0, cm, xA, xB, xC):
  o = np.zeros_like(x)
  # Cases, first and third case just keep zeros
  # Second case
  idx = np.array(((x >= xA(t)) & (x <= xB(t))))
  idx2 = np.array((t == 0)) # indexes where t == 0
  t[idx2] = 1
  o[idx] = 2 / 3 * ((x[idx] - x0) / t[idx] + c0)
  # Third case
  idx3 = np.array((x > xB(t)) & (x <= xC(t)))
  o[idx3] = 2 * (c0 - cm)
  # Fourth case
  idx_4 = np.array((x > xC(t)))
  o[idx_4] = 0
  
  return o

xA_ = lambda t, c0, x0: x0 - c0 * t
xB_ = lambda t, um, cm, x0: x0 + (um - cm) * t
xC_ = lambda t, vc, x0: x0 + vc * t
  
#%% Analytic
# Parameters
h_0 = 40
h_m = 18
h_d = 1
g = 1
x_0 = 1000
c_0 = np.sqrt(g * h_0)
c_m = np.sqrt(g * h_m)
c_d = np.sqrt(g * h_d)
u_m = 2 * (c_0 - c_m)
v_c = h_m * u_m / (h_m - h_d)

xA = lambda t: xA_(t, c_0, x_0)
xB = lambda t: xB_(t, u_m, c_m, x_0)
xC = lambda t: xC_(t, v_c, x_0)

h = lambda x, t: h_(x, t, h_0, h_m, h_d, x_0, g, c_0, xA, xB, xC)
u = lambda x, t: u_(x, t, x_0, c_0, c_m, xA, xB, xC)

x_i, x_f = 0, 2000
t_i, t_f = 0, 40
Nx = 201
Nt = 201
x = np.linspace(x_i, x_f, Nx)
t = np.linspace(t_i, t_f, Nt)
X, T = np.meshgrid(x, t)

#%% Ritter
HS = h(X, T)
US = u(X, T)

#%% Initial condition
plot1D(x, HS[0])

#%% End
plot1D(x, HS[-1])

#%% Evolution
plot2D(x, t, HS)

#%% 3D
plot3D(X, T, HS)

#%% Initial condition
plot1D(x, US[-1])

#%% Evolution
plot2D(x, t, US)

#%% 3D
plot3D(X, T, US)

#%% Numerical Stoker
h_0 = 40
h_d = 1
x_0 = 1000
L = 2000
T = 40 # 1
Nx = 200
Nt = 500 # 5000
f = 0
g = 1#9.8 #1

h0_ = lambda x, x_0, h_0, h_d: np.piecewise(x, [x <= x_0, x > x_0], [h_0, h_d]) 
h0 = lambda x: h0_(x, x_0, h_0, h_d) # Initial condition h
u0 = lambda x: x * 0  # Initial condition u
Sf = lambda f, g, h, Q: f * np.abs(Q) * Q / (8 * g * h ** 3) # Friction function
#%%
stoker = Experiment1D(
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
tl, xl, Hl, Ql = stoker.solvePDE('lf')

#%% Rusanov
tr, xr, Hr, Qr = stoker.solvePDE('rs')

#%%
k = -1
#%%
plot1D(xl, Hl[k])

#%%
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
compare(xl, HS[-1], Hl[-1], Hr[-1])
print("Error Lax-Friedrichs: ", np.linalg.norm(HS[-1] - Hl[-1]))
print("Error Rusanov: ", np.linalg.norm(HS[-1] - Hr[-1]))

print("u(x,t)")
compare(xl, US[-1], Ql[-1] / Hl[-1], Qr[-1] / Hr[-1])
print("Error Lax-Friedrichs: ", np.linalg.norm(US[-1] - Ql[-1] / Hl[-1]))
print("Error Rusanov: ", np.linalg.norm(US[-1] - Qr[-1] / Hr[-1]))

#%%Save data
DIR = 'data/3/2/' # Directory name
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True) # Create Folder
#%% Generate data

#Evolution of solution
M, N = 51, 51
data1 = np.zeros((M * N, 8)) # Evolution

Ul = Ql / Hl
Ur = Qr / Hr

data1[:,0] = XLF[::1,::4].flatten()
data1[:,1] = TLF[::1,::4].flatten()
data1[:,2] = HS[::4,::4].flatten()
data1[:,3] = Hl[::10,::4].flatten()
data1[:,4] = Hr[::10,::4].flatten()
data1[:,5] = US[::4,::4].flatten()
data1[:,6] = Ul[::10,::4].flatten()
data1[:,7] = Ur[::10,::4].flatten()

np.savetxt(DIR + 'stoker_1D.csv', data1, fmt='%.16f', delimiter=' ', header='x t a l r u ul ur', comments="") # Save data

#%%
data2 = np.zeros((201, 7)) # Last value

data2[:,0] = xr
data2[:,1] = HS[-1]
data2[:,2] = Hl[-1]
data2[:,3] = Hr[-1]
data2[:,4] = US[-1]
data2[:,5] = Ul[-1]
data2[:,6] = Ur[-1]

np.savetxt(DIR + 'compare_stoker_1D.csv', data2, fmt='%.16f', delimiter=' ', header='x h l r u ul ur', comments="") # Save data
