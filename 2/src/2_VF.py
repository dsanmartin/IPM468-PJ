import pathlib
import numpy as np
from atmospheric import Experiment
from plot import plot1D, plot2D, plot3D, quiver
  
#%% Initial conditions
h0 = lambda x, y, R, hp: 1 + hp * (np.sqrt((x - .5)**2 + (y - .5)**2) <= R) # Initial 
u0 = lambda x, y: x * 0
v0 = lambda x, y: x * 0

#%% Parameters

# Model parameters
H_ = .5
f_ = 0
g_ = 1
b_ = 2

# Domain limits
xf = 1 # "L"
yf = 1
tf = 1

# Domain grid size
Nt = 500
Nx = 100
Ny = Nx

bc = 1

# Create experiment
exp = Experiment(
  H = H_,
  f = f_,
  g = g_,
  b = b_,
  Nx = Nx,
  Ny = Ny,
  Nt = Nt,
  xf = xf,
  yf = yf,
  tf = tf,
  bc = bc,
  h0 = lambda x, y: h0(x, y, .1, .1),
  u0 = u0,
  v0 = v0,
)

#%% QUESTION 2.1 ###
tr, Xr, Yr, Hr, Ur, Vr = exp.solveVF('rk4')

#%% Plot exp 1
plot1D(tr, Hr[:, Ny//2, Nx//2])

#%%
n = 250
# for n in range(len(tr)):
#   if n % 100 == 0: 
#     print(n)
plot2D(Xr, Yr, Hr[n])
    
#%%
plot3D(Xr, Yr, Hr[n])

#%%
quiver(Xr, Yr, Ur[125], Vr[125])

  #%% QUESTION 2.2 - Coriolis effect %%
Om = 7.2921e-5 # Angular speed
phi = -33.036 * np.pi / 180 # Valparaiso Latitude
f1 = 2 * Om * np.sin(phi)
exp.H = .5 # Selected H
#%% Simulation
exp.f = f1 * 1e5
tf1, Xf1, Yf1, Hf1, Uf1, Vf1 = exp.solveVF('rk4')

#%%
plot1D(tf1, Hf1[:, Ny//2, Nx//2])
  
#%%
#n = -1
#XX,  = np.meshgrid(Xf1[0], tf1)
for n in range(Nt):
  if n % 50 == 0:
    print(n)
    plot3D(Xf1, Yf1, Hf1[n])

#%%
for n in range(Nt):
  if n % 50 == 0:
    print(n)
    quiver(Xf1, Yf1, Uf1[n], Vf1[n])

#%% Save data for plots question 1
DIR = 'data/2/1/' # Directory name
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True) # Create Folder
#%%
data_t = np.zeros((Nt + 1, 2))
data_t[:,0] = tr
data_t[:,1] = Hr[:, Ny // 2, Nx // 2]

np.savetxt(DIR + 'ht_vf.csv', data_t, fmt='%.16f', delimiter=',', header='t,h', comments="") # Save data

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

np.savetxt(DIR + 'sv_2D_vf.csv', data_h, fmt='%.16f', delimiter=' ', header='x y h0 h10 h20 h30 h40', comments="") # Save data

#%%

nn = 3
MM, NN = Ur[0, ::nn, ::nn].shape
data_v = np.zeros((MM * NN, 17))
data_v[:, 0] = Xr[::nn, ::nn].flatten()
data_v[:, 1] = Yr[::nn, ::nn].flatten()
data_v[:, 2] = Ur[0, ::nn, ::nn].flatten() #/ Hr[0, ::nn, ::nn].flatten()
data_v[:, 3] = Vr[0, ::nn, ::nn].flatten() #/ Hr[0, ::nn, ::nn].flatten()
data_v[:, 4] = np.sqrt(data_v[:, 2] ** 2 + data_v[:, 3] ** 2)
data_v[:, 5] = Ur[125, ::nn, ::nn].flatten() #/ Hr[125, ::nn, ::nn].flatten()
data_v[:, 6] = Vr[125, ::nn, ::nn].flatten() #/ Hr[125, ::nn, ::nn].flatten()
data_v[:, 7] = np.sqrt(data_v[:, 5] ** 2 + data_v[:, 6] ** 2)
data_v[:, 8] = Ur[250, ::nn, ::nn].flatten() #/ Hr[250, ::nn, ::nn].flatten()
data_v[:, 9] = Vr[250, ::nn, ::nn].flatten() #/ Hr[250, ::nn, ::nn].flatten()
data_v[:, 10] = np.sqrt(data_v[:, 8] ** 2 + data_v[:, 9] ** 2)
data_v[:, 11] = Ur[375, ::nn, ::nn].flatten() #/ Hr[375, ::nn, ::nn].flatten()
data_v[:, 12] = Vr[375, ::nn, ::nn].flatten() #/ Hr[375, ::nn, ::nn].flatten()
data_v[:, 13] = np.sqrt(data_v[:, 11] ** 2 + data_v[:, 12] ** 2)
data_v[:, 14] = Ur[-1, ::nn, ::nn].flatten() #/ Hr[-1, ::nn, ::nn].flatten()
data_v[:, 15] = Vr[-1, ::nn, ::nn].flatten() #/ Hr[-1, ::nn, ::nn].flatten()
data_v[:, 16] = np.sqrt(data_v[:, 14] ** 2 + data_v[:, 15] ** 2)

header_ = 'x y u_0 v_0 m0 u_10 v_10 m10 u_20 v_20 m20 u_30 v_30 m30 u_40 v_40 m40'
np.savetxt(DIR + 'sv_v_2D_vf.csv', data_v, fmt='%.16f', delimiter=' ', header=header_, comments="")


#%%
DIR2 = 'data/2/2/' # Directory name
pathlib.Path(DIR2).mkdir(parents=True, exist_ok=True) # Create Folder
#%%
data_tc = np.zeros((Nt + 1, 2))
data_tc[:,0] = tf1
data_tc[:,1] = Hf1[:, Ny // 2, Nx // 2]

np.savetxt(DIR2 + 'ht_c_vf.csv', data_tc, fmt='%.16f', delimiter=',', header='t,h', comments="") # Save data


#%% Save experiment n = {0, 125, 250, 375, -1}
M, N = Hf1[0].shape

data_hf = np.zeros((M * N, 7))
data_hf[:, 0] = Xf1.flatten()
data_hf[:, 1] = Yf1.flatten()
data_hf[:, 2] = Hf1[0].flatten()
data_hf[:, 3] = Hf1[125].flatten()
data_hf[:, 4] = Hf1[250].flatten()
data_hf[:, 5] = Hf1[375].flatten()
data_hf[:, 6] = Hf1[-1].flatten()

np.savetxt(DIR2 + 'sv_c_2D_vf.csv', data_hf, fmt='%.16f', delimiter=' ', header='x y h0 h10 h20 h30 h40', comments="") # Save data

#%%

nn = 3
MM, NN = Uf1[0, ::nn, ::nn].shape
data_vf = np.zeros((MM * NN, 17))
data_vf[:, 0] = Xf1[::nn, ::nn].flatten()
data_vf[:, 1] = Yf1[::nn, ::nn].flatten()
data_vf[:, 2] = Uf1[0, ::nn, ::nn].flatten() #/ Hr[0, ::nn, ::nn].flatten()
data_vf[:, 3] = Vf1[0, ::nn, ::nn].flatten() #/ Hr[0, ::nn, ::nn].flatten()
data_vf[:, 4] = np.sqrt(data_vf[:, 2] ** 2 + data_vf[:, 3] ** 2)
data_vf[:, 5] = Uf1[125, ::nn, ::nn].flatten() #/ Hr[125, ::nn, ::nn].flatten()
data_vf[:, 6] = Vf1[125, ::nn, ::nn].flatten() #/ Hr[125, ::nn, ::nn].flatten()
data_vf[:, 7] = np.sqrt(data_vf[:, 5] ** 2 + data_vf[:, 6] ** 2)
data_vf[:, 8] = Uf1[250, ::nn, ::nn].flatten() #/ Hr[250, ::nn, ::nn].flatten()
data_vf[:, 9] = Vf1[250, ::nn, ::nn].flatten() #/ Hr[250, ::nn, ::nn].flatten()
data_vf[:, 10] = np.sqrt(data_vf[:, 8] ** 2 + data_vf[:, 9] ** 2)
data_vf[:, 11] = Uf1[375, ::nn, ::nn].flatten() #/ Hr[375, ::nn, ::nn].flatten()
data_vf[:, 12] = Vf1[375, ::nn, ::nn].flatten() #/ Hr[375, ::nn, ::nn].flatten()
data_vf[:, 13] = np.sqrt(data_vf[:, 11] ** 2 + data_vf[:, 12] ** 2)
data_vf[:, 14] = Uf1[-1, ::nn, ::nn].flatten() #/ Hr[-1, ::nn, ::nn].flatten()
data_vf[:, 15] = Vf1[-1, ::nn, ::nn].flatten() #/ Hr[-1, ::nn, ::nn].flatten()
data_vf[:, 16] = np.sqrt(data_vf[:, 14] ** 2 + data_vf[:, 15] ** 2)

header_ = 'x y u_0 v_0 m0 u_10 v_10 m10 u_20 v_20 m20 u_30 v_30 m30 u_40 v_40 m40'
np.savetxt(DIR2 + 'sv_c_v_2D_vf.csv', data_vf, fmt='%.16f', delimiter=' ', header=header_, comments="")

