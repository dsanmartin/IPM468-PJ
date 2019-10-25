"""
  Question 3 prototype (only testing...)
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
# 
def F2(t, u):
  return np.array([
    -(u[1] - u[3]) / r(u[:2], u[2:]) ** 2,
    (u[0] - u[2]) / r(u[:2], u[2:]) ** 2,
    -(u[3] - u[1]) / r(u[2:], u[:2]) ** 2,
    (u[2] - u[0]) / r(u[2:], u[:2]) ** 2,
  ]) / (2 * np.pi)

def F(t, u):
  return np.array([
    -(u[1] - u[3]) / r(u[0:2], u[2:4]) ** 2 - (u[1] - u[5]) / r(u[0:2], u[4:6]) ** 2 - (u[1] - u[7]) / r(u[0:2], u[6:]) ** 2, 
    (u[0] - u[2]) / r(u[0:2], u[2:4]) ** 2 + (u[0] - u[4]) / r(u[0:2], u[4:6]) ** 2 + (u[0] - u[6]) / r(u[0:2], u[6:]) ** 2, 
    -(u[3] - u[1]) / r(u[2:4], u[0:2]) ** 2 - (u[3] - u[5]) / r(u[2:4], u[4:6]) ** 2 - (u[3] - u[7]) / r(u[2:4], u[6:]) ** 2,
    (u[2] - u[0]) / r(u[2:4], u[0:2]) ** 2 + (u[2] - u[4]) / r(u[2:4], u[4:6]) ** 2 + (u[2] - u[6]) / r(u[2:4], u[6:]) ** 2,
    -(u[5] - u[1]) / r(u[4:6], u[0:2]) ** 2 - (u[5] - u[3]) / r(u[4:6], u[2:4]) ** 2 - (u[5] - u[7]) / r(u[4:6], u[6:]) ** 2,
    (u[4] - u[0]) / r(u[4:6], u[0:2]) ** 2 + (u[4] - u[2]) / r(u[4:6], u[2:4]) ** 2 + (u[4] - u[6]) / r(u[4:6], u[6:]) ** 2,
    -(u[7] - u[1]) / r(u[6: ], u[0:2]) ** 2 - (u[7] - u[3]) / r(u[6: ], u[2:4]) ** 2 - (u[7] - u[5]) / r(u[6: ], u[4:6]) ** 2,
    (u[6] - u[0]) / r(u[6: ], u[0:2]) ** 2 + (u[6] - u[2]) / r(u[6: ], u[2:4]) ** 2 + (u[6] - u[4]) / r(u[6: ], u[4:6]) ** 2,
  ]) / (2 * np.pi)
  
  
def RK4(t, u0, fun):
  L = len(t)
  U = np.zeros((L, len(u0)))
  U[0] = u0
  dt = t[1] - t[0]
  for k in range(L - 1):
    k1 = fun(t[k], U[k])
    k2 = fun(t[k] + 0.5 * dt, U[k] + 0.5 * dt * k1)
    k3 = fun(t[k] + 0.5 * dt, U[k] + 0.5 * dt * k2)
    k4 = fun(t[k] + dt, U[k] + dt * k3)
    U[k + 1] = U[k] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

  return U  

def r(p1, p2):
  (x1, y1), (x2, y2) = p1, p2
  return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def CE(p):
  rij = 1
  N = len(p) // 2
  for j in range(N):
    for i in range(N):
      if i != j:
        p1 = p[2*i:2*(i+1)]
        p2 = p[2*j:2*(j+1)]
        rij *= np.sqrt(r(p1, p2))
  return rij

def R(U):
  return np.array([CE(U[k]) for k in range(len(U))])

def RR(P1, P2):
  return np.array([r(P1[k], P2[k]) for k in range(len(P1))])

def plot(U):
  M, N = U.shape
  for p in range(0, N, 2):
    plt.scatter(U[:, p], U[:, p + 1])
  plt.grid(True)
  plt.show()
  
def plotR(t, r):
  plt.plot(t, r)
  plt.grid(True)
  plt.show()
#%% Parameters
L = 500
T_max = 500
t_1 = np.linspace(0, T_max, L + 1)
# %% P1
u0_1 = np.array([1, 1, -1, 1, -1, -1, 1, -1])
U1 = RK4(t_1, u0_1, F)
#%%
plot(U1)
# %% P2
u0_2 = np.array([1, 1.01, -1, 1, -1, -1, 1, -1])
U2 = RK4(t_1, u0_2, F)
#%%
plot(U2)
# %% P3
u0_3 = np.array([2, 1, -2, 1, -2, -1, 2, -1])
U3 = RK4(t_1, u0_3, F)
#%%
plot(U3)

# %% P4
u0_4 = np.array([2, 1.01, -2, 1, -2, -1, 2, -1])
U4 = RK4(t_1, u0_4, F)
#%%
plot(U4)

# %% P5
L = 500
T_max = 200
t_2 = np.linspace(0, T_max, L + 1)
#%% a
eps = 0
u0_5a = np.array([-1, 0, eps, 0, 1, 0, 2, 0])
U5a = RK4(t_2, u0_5a, F)
#%%
plot(U5a)

#%% b
eps = 1e-4
u0_5b = np.array([-1, 0, eps, 0, 1, 0, 2, 0])
U5b = RK4(t_2, u0_5b, F)
#%%
plot(U5b)
#%%
R1 = R(U1)
R2 = R(U2)
R3 = R(U3)
R4 = R(U4)
R5a = R(U5a)
R5b = R(U5b)
#%%
plotR(t_1, R1)
plotR(t_1, R2)
plotR(t_1, R3)
plotR(t_1, R4)
plotR(t_2, R5a)
plotR(t_2, R5b)
#%%
#R5a_1 = R2()
#plt.plot()

#%%
RT1 = np.zeros((len(R1), 2)); RT1[:,0] = t_1; RT1[:,1] = R1
RT2 = np.zeros((len(R2), 2)); RT2[:,0] = t_1; RT2[:,1] = R2
RT3 = np.zeros((len(R3), 2)); RT3[:,0] = t_1; RT3[:,1] = R3
RT4 = np.zeros((len(R4), 2)); RT4[:,0] = t_1; RT4[:,1] = R4
RT5a = np.zeros((len(R5a), 2)); RT5a[:,0] = t_2; RT5a[:,1] = R5a
RT5b = np.zeros((len(R5b), 2)); RT5b[:,0] = t_2; RT5b[:,1] = R5b
#%% SAVE
DIR = 'data/3/'
np.savetxt(DIR + 'U1.csv', U1, fmt='%.8f', delimiter=',', header='x1,y1,x2,y2,x3,y3,x4,y4', comments="")
np.savetxt(DIR + 'U2.csv', U2, fmt='%.8f', delimiter=',', header='x1,y1,x2,y2,x3,y3,x4,y4', comments="")
np.savetxt(DIR + 'U3.csv', U3, fmt='%.8f', delimiter=',', header='x1,y1,x2,y2,x3,y3,x4,y4', comments="")
np.savetxt(DIR + 'U4.csv', U4, fmt='%.8f', delimiter=',', header='x1,y1,x2,y2,x3,y3,x4,y4', comments="")
np.savetxt(DIR + 'U5a.csv', U5a, fmt='%.8f', delimiter=',', header='x1,y1,x2,y2,x3,y3,x4,y4', comments="")
np.savetxt(DIR + 'U5b.csv', U5b, fmt='%.8f', delimiter=',', header='x1,y1,x2,y2,x3,y3,x4,y4', comments="")
np.savetxt(DIR + 'R1.csv', RT1, fmt='%.8f', delimiter=',', header='t,r', comments="")
np.savetxt(DIR + 'R2.csv', RT2, fmt='%.8f', delimiter=',', header='t,r', comments="")
np.savetxt(DIR + 'R3.csv', RT3, fmt='%.8f', delimiter=',', header='t,r', comments="")
np.savetxt(DIR + 'R4.csv', RT4, fmt='%.8f', delimiter=',', header='t,r', comments="")
np.savetxt(DIR + 'R5a.csv', RT5a, fmt='%.8f', delimiter=',', header='t,r', comments="")
np.savetxt(DIR + 'R5b.csv', RT5b, fmt='%.8f', delimiter=',', header='t,r', comments="")