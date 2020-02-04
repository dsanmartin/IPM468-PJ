import numpy as np
import matplotlib.pyplot as plt
#%%

h0 = 5
g = 1

c0 = np.sqrt(g * h0)

xA = lambda t: -c0 * t
xB = lambda t: 2 * c0 * t
  

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

#def h(x, t):
#  if type(t) is int:
#    return h_(x,t)
#  else:
#    o = np.zeros((len(t), len(x)))
#    for k in range(len(t)):
#      o[k] = h_(x, t[k])
#  return o
#
#def u(x, t):
#  if type(t) is int:
#    return u_(x,t)
#  else:
#    o = np.zeros((len(t), len(x)))
#    for k in range(len(t)):
#      o[k] = u_(x, t[k])
#  return o
  
#%%
x_i, x_f = -10, 10
t_i, t_f = 0, 5
Nx = 500
Nt = 100
x = np.linspace(x_i, x_f, Nx)
t = np.linspace(t_i, t_f, Nt)
X, T = np.meshgrid(x, t)

#%% Initial condition

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(x, h_(x, T[0]))
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(x, u_(x, T[0]))
plt.grid(True)
plt.show()


#%% Evolution
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.contourf(x, t, h_(X, T))
plt.grid(True)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.contourf(x, t, u_(X, T))
plt.grid(True)
plt.colorbar()
plt.show()

#%%
from mpl_toolkits import mplot3d

#fig, (ax1, ax2) = plt.subplots(2, 1)
fig = plt.figure(figsize=(8, 3))
#plt.subplot(1, 2, 1)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, T, h_(X, T))
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, T, u_(X, T))
plt.show()

