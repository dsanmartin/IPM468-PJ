import pathlib
import numpy as np
from dambreak import Experiment2D
from plot import quiver, plot2D, plot3D

#%%
def h0_(x, y, x0, y0, h0):
  H = np.zeros_like(x) #* 0# 1e-16
  idx = np.array((x <= x0) & (y <= y0))
  H[idx] = h0
  return H
  
#%%
h_0 = 40
x0 = 1000
y0 = 1000
L = 2000
T = 40
Nx = 100
Ny = 100
Nt = 500
g = 1
f = 8 * g / (40 ** 2)

h0 = lambda x, y: h0_(x, y, x0, y0, h_0) 
u0 = lambda x, y: x * 0
v0 = lambda x, y: y * 0
Sf = lambda f, g, h, Q: f * np.abs(Q) * Q / (8 * g * h ** 3)

#%%
dressler = Experiment2D(
  f = f,
  g = g,
  L = L,
  T = T,
  Nx = Nx,
  Ny = Ny,
  Nt = Nt,
  h0 = h0,
  u0 = u0,
  v0 = v0,
  Sf = Sf
)

#%% Lax-Friedrich scheme not working...
t, Xl, Yl, Hl, Q1l, Q2l = dressler.solvePDE('lf')

#%%
t, Xr, Yr, Hr, Q1r, Q2r = dressler.solvePDE('rs')

#%%
n = 5
plot3D(Xl, Yl, Hl[n])

#%%
#plot2D(x, t, H[1])
n = -1
plot2D(Xr, Yr, Hr[n])

#%%
n = -1
plot3D(Xr, Yr, Hr[n])

#%%
quiver(Xr, Yr, Q1r[n], Q2r[n])

#%%Save data
DIR = 'data/3/3/' # Directory name
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True) # Create Folder

#%% Save experiment n = {0, 125, 250, 375, -1}
M, N = Hr[0].shape

data_h = np.zeros((M * N, 7))
data_h[:, 0] = Xr.flatten()
data_h[:, 1] = Yr.flatten()
data_h[:, 2] = Hr[0].flatten()
data_h[:, 3] = Hr[125].flatten()
data_h[:, 4] = Hr[250].flatten()
data_h[:, 5] = Hr[375].flatten()
data_h[:, 6] = Hr[-1].flatten()

np.savetxt(DIR + 'dressler_2D.csv', data_h, fmt='%.16f', delimiter=' ', header='x y h0 h10 h20 h30 h40', comments="") # Save data

#%%

nn = 3
MM, NN = Q1r[0, ::nn, ::nn].shape
data_v = np.zeros((MM * NN, 17))
data_v[:, 0] = Xr[::nn, ::nn].flatten()
data_v[:, 1] = Yr[::nn, ::nn].flatten()
data_v[:, 2] = Q1r[0, ::nn, ::nn].flatten() / Hr[0, ::nn, ::nn].flatten()
data_v[:, 3] = Q2r[0, ::nn, ::nn].flatten() / Hr[0, ::nn, ::nn].flatten()
data_v[:, 4] = np.sqrt(data_v[:, 2] ** 2 + data_v[:, 3] ** 2)
data_v[:, 5] = Q1r[125, ::nn, ::nn].flatten() / Hr[125, ::nn, ::nn].flatten()
data_v[:, 6] = Q2r[125, ::nn, ::nn].flatten() / Hr[125, ::nn, ::nn].flatten()
data_v[:, 7] = np.sqrt(data_v[:, 5] ** 2 + data_v[:, 6] ** 2)
data_v[:, 8] = Q1r[250, ::nn, ::nn].flatten() / Hr[250, ::nn, ::nn].flatten()
data_v[:, 9] = Q2r[250, ::nn, ::nn].flatten() / Hr[250, ::nn, ::nn].flatten()
data_v[:, 10] = np.sqrt(data_v[:, 8] ** 2 + data_v[:, 9] ** 2)
data_v[:, 11] = Q1r[375, ::nn, ::nn].flatten() / Hr[375, ::nn, ::nn].flatten()
data_v[:, 12] = Q2r[375, ::nn, ::nn].flatten() / Hr[375, ::nn, ::nn].flatten()
data_v[:, 13] = np.sqrt(data_v[:, 11] ** 2 + data_v[:, 12] ** 2)
data_v[:, 14] = Q1r[-1, ::nn, ::nn].flatten() / Hr[-1, ::nn, ::nn].flatten()
data_v[:, 15] = Q2r[-1, ::nn, ::nn].flatten() / Hr[-1, ::nn, ::nn].flatten()
data_v[:, 16] = np.sqrt(data_v[:, 14] ** 2 + data_v[:, 15] ** 2)

header_ = 'x y u_0 v_0 m0 u_10 v_10 m10 u_20 v_20 m20 u_30 v_30 m30 u_40 v_40 m40'
np.savetxt(DIR + 'dressler_v_2D.csv', data_v, fmt='%.16f', delimiter=' ', header=header_, comments="")

#%%
MMM, NNN = Hl[0].shape

data_lf = np.zeros((MMM * NNN, 3))
data_lf[:, 0] = Xl.flatten()
data_lf[:, 1] = Yl.flatten()
data_lf[:, 2] = Hl[4].flatten()

np.savetxt(DIR + "dressler_2D_lf.csv", data_lf, fmt='%.16f', delimiter=' ', header="x y h", comments="")









