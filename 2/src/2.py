import pathlib
import numpy as np
from atmospheric import Experiment
from plot import plot1D, plot2D, plot3D
  
#%%
def airy(x, t, H, k, w):
  return .5 * H * np.cos(k * x - w * t)
#%%
# Initial conditions
h0 = lambda x, y, R, hp: 1 + hp * (np.sqrt((x - .5)**2 + (y - .5)**2) <= R) # Initial 
u0 = lambda x, y: x * 0
#h0g = lambda x, y: 1 + 0.11 * np.exp(-100*((x-.5)**2 + (y-.5)**2))
v0 = lambda x, y: x * 0

#%%

# Model parameters
H_ = .1
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
  #h0 = h0g,
  u0 = u0,
  v0 = v0,
)

#%% QUESTION 2.1 - Experiments changing H value ###

#%% Exp 1
exp.H = .1
t1, X1, Y1, H1, U1, V1 = exp.solvePDE()

#%% Plot exp 1
plot1D(t1, H1[:, Ny//2, Nx//2])

#%% Exp 2
exp.H = .5
t2, X2, Y2, H2, U2, V2 = exp.solvePDE()

#%% Plot exp 2
plot1D(t2, H2[:, Ny//2, Nx//2])


#%% Exp 3
exp.H = 1
t3, X3, Y3, H3, U3, V3 = exp.solvePDE()

#%% Plot exp 3
plot1D(t3, H3[:, Ny//2, Nx//2])

#%% Exp 4
exp.H = 2
t4, X4, Y4, H4, U4, V4 = exp.solvePDE()

#%% Plot exp 1
plot1D(t4, H4[:, Ny//2, Nx//2])

#%%
for k in range(len(t1)):
  if k % 100 == 0: 
    plot2D(X1, Y1, H1[k])
    
#%%
plot3D(X1, Y1, H1[-1])

#%%
N = 500
x = np.linspace(0, xf, N)
t = np.linspace(0, tf, N)
l = .75
T = .055
k = 2 * np.pi / l
w = 2 * np.pi / T
eta = 1 + airy(x, t, .05, k, w)

#%%
#plot1D(t, eta)
import matplotlib.pyplot as plt
plt.plot(t2, H2[:, Ny//2, Nx//2])
plt.plot(t, eta)
plt.grid(True)
plt.show()

#%%

#%%
plot1D(x, eta)
# #%%
# XX, TT = np.meshgrid(x,t)
# Z = airy(XX, TT, 1, k, w)
# #%%
# plt.contourf(XX, TT, Z)
# plt.show()


#%% QUESTION 2.2 - Coriolis effect %%
Om = 7.2921e-5 # Angular speed
phi = -33.036 * np.pi / 180 # Valparaiso Latitude
f1 = 2 * Om * np.sin(phi)
exp.f = f1
exp.H = 1 # Selected H
#%%
exp.f = f1 / 1000
tf1, Xf1, Yf1, Hf1, Uf1, Vf1 = exp.solvePDE()
#%%
plot1D(tf1, Hf1[:, Ny//2, Nx//2])



#%% Save data for plots
DIR = 'data/2/1/' # Directory name
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True) # Create Folder
#%% Generate data
data1 = np.zeros((Nt + 1, 2))
data2 = np.zeros((Nt + 1, 2))
data3 = np.zeros((Nt + 1, 2))
data4 = np.zeros((Nt + 1, 2))
data1[:,0] = t1
data1[:,1] = H1[:, Ny // 2, Nx // 2].flatten()
data2[:,0] = t2
data2[:,1] = H2[:, Ny // 2, Nx // 2].flatten()
data3[:,0] = t3
data3[:,1] = H3[:, Ny // 2, Nx // 2].flatten()
data4[:,0] = t4
data4[:,1] = H4[:, Ny // 2, Nx // 2].flatten()

#%%
np.savetxt(DIR + 'h1.csv', data1, fmt='%.16f', delimiter=',', header='t,h', comments="") # Save data
np.savetxt(DIR + 'h2.csv', data2, fmt='%.16f', delimiter=',', header='t,h', comments="") # Save data
np.savetxt(DIR + 'h3.csv', data3, fmt='%.16f', delimiter=',', header='t,h', comments="") # Save data
np.savetxt(DIR + 'h4.csv', data4, fmt='%.16f', delimiter=',', header='t,h', comments="") # Save data

#%% Save experiment complete

M, N = X1.shape
sim_t0 = np.zeros((M * N, 5))
sim_t1 = np.zeros((M * N, 5))
sim_t2 = np.zeros((M * N, 5))
sim_t3 = np.zeros((M * N, 5))
sim_t4 = np.zeros((M * N, 5))

sim_t0[:, 0] = X2.flatten(); sim_t0[:, 1] = Y2.flatten(); sim_t0[:, 2] = H2[  0].flatten(); sim_t0[:, 3] = U2[  0].flatten(); sim_t0[:, 4] = V2[  0].flatten()
sim_t1[:, 0] = X2.flatten(); sim_t1[:, 1] = Y2.flatten(); sim_t1[:, 2] = H2[125].flatten(); sim_t0[:, 3] = U2[125].flatten(); sim_t0[:, 4] = V2[125].flatten()
sim_t2[:, 0] = X2.flatten(); sim_t2[:, 1] = Y2.flatten(); sim_t2[:, 2] = H2[250].flatten(); sim_t0[:, 3] = U2[250].flatten(); sim_t0[:, 4] = V2[250].flatten()
sim_t3[:, 0] = X2.flatten(); sim_t3[:, 1] = Y2.flatten(); sim_t3[:, 2] = H2[375].flatten(); sim_t0[:, 3] = U2[375].flatten(); sim_t0[:, 4] = V2[375].flatten()
sim_t4[:, 0] = X2.flatten(); sim_t4[:, 1] = Y2.flatten(); sim_t4[:, 2] = H2[ -1].flatten(); sim_t0[:, 3] = U2[ -1].flatten(); sim_t0[:, 4] = V2[ -1].flatten()

np.savetxt(DIR + 'sim_t0.csv', sim_t0, fmt='%.16f', delimiter=',', header='x,y,h,u,v', comments="") # Save data
np.savetxt(DIR + 'sim_t1.csv', sim_t1, fmt='%.16f', delimiter=',', header='x,y,h,u,v', comments="") # Save data
np.savetxt(DIR + 'sim_t2.csv', sim_t2, fmt='%.16f', delimiter=',', header='x,y,h,u,v', comments="") # Save data
np.savetxt(DIR + 'sim_t3.csv', sim_t3, fmt='%.16f', delimiter=',', header='x,y,h,u,v', comments="") # Save data
np.savetxt(DIR + 'sim_t4.csv', sim_t4, fmt='%.16f', delimiter=',', header='x,y,h,u,v', comments="") # Save data


#%% Experiments f1
cor = np.zeros((len(tf1), 7))
cor[:,0] = tf1
cor[:,1] = Hf1[:, Ny//2, Nx//2]
cor[:,2] = Hf1[:, Ny//2, Nx//2]
cor[:,3] = Hf1[:, Ny//2, Nx//2]
cor[:,4] = Hf1[:, Ny//2, Nx//2]
cor[:,5] = Hf1[:, Ny//2, Nx//2]
cor[:,6] = Hf1[:, Ny//2, Nx//2]

np.savetxt(DIR + 'coriolis.csv', cor, fmt='%.16f', delimiter=' ', header='t f1 f2 f3 f4 f5 f6', comments="") # Save data




















#%%
import matplotlib.pyplot as plt
nn=2
fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
#axs[0].imshow(H2[0], extent=[0, 1, 0, 1])
#axs[0].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[0,::nn,::nn], V2[0,::nn,::nn])
axs[0,0].imshow(H2[125], extent=[0, 1, 0, 1])
axs[0,0].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[125,::nn,::nn], V2[125,::nn,::nn])
axs[0,1].imshow(H2[250], extent=[0, 1, 0, 1])
axs[0,1].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[250,::nn,::nn], V2[250,::nn,::nn])
axs[1,0].imshow(H2[375], extent=[0, 1, 0, 1])
axs[1,0].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[375,::nn,::nn], V2[375,::nn,::nn])
axs[1,1].imshow(H2[ -1], extent=[0, 1, 0, 1])
axs[1,1].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[ -1,::nn,::nn], V2[ -1,::nn,::nn])

for ax in axs.flat:
    ax.set(xlabel=r'$x$', ylabel=r'$y$')
    
for ax in axs.flat:
    ax.label_outer()
    
plt.show()

#%%
nn=3
MM, NN = X1[::nn,::nn].shape 
test = np.zeros((MM*NN, 5))
kk = -1
test[:,0] = X2[::nn,::nn].flatten()
test[:,1] = Y2[::nn,::nn].flatten()
test[:,2] = U2[kk,::nn,::nn].flatten()
test[:,3] = V2[kk,::nn,::nn].flatten()
test[:,4] = np.sqrt(test[:,2]**2 + test[:,3]**2)

np.savetxt(DIR + 'test4.csv', test, fmt='%.16f', delimiter=' ', header='x y u v m', comments="") # Save data

#%%
import matplotlib.pyplot as plt
nn=2
fig, axs = plt.subplots(5, 2, figsize=(6, 6), sharex=True, sharey=True)
axs[0, 0].imshow(np.sqrt(U1[0] ** 2 + V1[0] ** 2), extent=[0, 1, 0, 1])
axs[0, 0].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[0,::nn,::nn], V2[0,::nn,::nn])
axs[0,1] = plt.gca(projection='3d')
axs[0,1].plot_surface(X2, Y2, H2[0])
#axs[0, 1].imshow(H2[125], extent=[0, 1, 0, 1])


# axs[0, 0].imshow(H2[125], extent=[0, 1, 0, 1])
# axs[0, 0].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[125,::nn,::nn], V2[125,::nn,::nn])
# axs[0, 1].imshow(H2[250], extent=[0, 1, 0, 1])
# axs[0, 1].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[250,::nn,::nn], V2[250,::nn,::nn])
# axs[1, 0].imshow(H2[375], extent=[0, 1, 0, 1])
# axs[1, 0].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[375,::nn,::nn], V2[375,::nn,::nn])
# axs[1, 1].imshow(H2[ -1], extent=[0, 1, 0, 1])
# axs[1, 1].quiver(X2[::nn,::nn], Y2[::nn,::nn], U2[ -1,::nn,::nn], V2[ -1,::nn,::nn])

for ax in axs.flat:
    ax.set(xlabel=r'$x$', ylabel=r'$y$')
    
for ax in axs.flat:
    ax.label_outer()
    
plt.show()




