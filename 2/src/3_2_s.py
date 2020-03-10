import numpy as np
import matplotlib.pyplot as plt
from embalse import Experiment1D
#%%

h0 = 4
hm = 2
hd = 1

x0 = 0
g = 1

c0 = np.sqrt(g * h0)
cm = np.sqrt(g * hm)
cd = np.sqrt(g * hd)

um = 2 * (c0 - cm)

vc = hm * um / (hm - hd)

xA = lambda t: -c0 * t
xB = lambda t: (um - cm) * t
xC = lambda t: vc * t

def h_0(x):
  o = np.zeros_like(x) + hd # Fill condition 2
  idx_1 = (x <= x0) # Index where condition 1 is 
  o[idx_1] = h0 
  return o


def h(x, t):
  # Data to return
  o = np.zeros_like(x)
  # Cases 
  # First case
  idx_1 = np.array((x <= xA(t))) # Index where condition 1 is 
  o[idx_1] = h0
  # Second case
  idx_2 = np.array((x >= xA(t)) & (x <= xB(t)))
  o[idx_2] = (-x[idx_2]/t[idx_2] + 2 * c0) ** 2 / (9 * g)
  # Third case
  idx_3 = np.array((x >= xB(t)) & (x <= xC(t)))
  o[idx_3] = hm
  # Fourth case
  idx_4 = np.array((x >= xC(t)))
  o[idx_4] = hd
  
  return o

  
def u(x, t):
  o = np.zeros_like(x)
  # Cases, first and third case just keep zeros
  # Second case
  idx = np.array(((x >= xA(t)) & (x <= xB(t))))
  o[idx] = 2 / 3 * (x[idx] / t[idx] + c0)
  
  return o
  
#%%
x = np.linspace(-5, 5, 500)
t = np.linspace(0, 5, 51)
X, T = np.meshgrid(x, t)

#%% Initial condition

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(x, h_0(x))
plt.grid(True)
#plt.subplot(1, 2, 2)
#plt.plot(x, u(x, T[0]))
#plt.grid(True)
plt.show()
#%%
x0 = 1000
L = 2000
T = 10
Nx = 100
Nt = 5000
f = 5#0
g = 9.8#1

#h0 = lambda x: h_0(x)#np.piecewise(x, [x < x0, x >= x0], [40, 1]) 
u0 = lambda x: x * 0
Sf = lambda f, g, h, Q: f * np.abs(Q) * Q / (8 * g * h ** 3)
#%%
exp_2 = Experiment1D(
  f = f,
  g = g,
  L = L,
  T = T,
  Nx = Nx,
  Nt = Nt,
  h0 = h_0,
  u0 = u0,
  Sf = Sf
)

#%%
t, x, H, Q = exp_2.solvePDE('lf')

#%%
# tt = 2
# k = np.where(t == tt)[0]
# print(k)
# H = h(X, T)[k].flatten()
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(x, H[1])
plt.grid(True)
#plt.subplot(1, 2, 2)
#plt.plot(x, u(x, T[0]))
#plt.grid(True)
plt.show()


#%% Evolution
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
#plt.contourf(x, t, h(X, T))
plt.contourf(x, t, H)
plt.grid(True)
plt.colorbar()
#plt.subplot(1, 2, 2)
#plt.contourf(x, t, u_(X, T))
#plt.grid(True)
#plt.colorbar()
plt.show()