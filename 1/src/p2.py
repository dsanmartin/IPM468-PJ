"""
  Question 2
"""
# Library import
import numpy as np # Data structures, vector, matrices, etc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt # Plots
from mpl_toolkits.mplot3d import Axes3D
#%%  
# RHS definition
def F(t, u, **kwargs):
  a, b, c = kwargs['a'], kwargs['b'], kwargs['c'] 
  return np.array([-u[1] - u[2], u[0] + a * u[1], b + u[2] * (u[0] - c)])

# Fourth-order Runge Kutta method
def RK4(t, u0, **kwargs):
  L = len(t)
  U = np.zeros((L, len(u0)))
  U[0] = u0
  dt = t[1] - t[0]
  for k in range(L-1):
    k1 = F(t[k], U[k], **kwargs)
    k2 = F(t[k] + 0.5 * dt, U[k] + 0.5 * dt * k1, **kwargs)
    k3 = F(t[k] + 0.5 * dt, U[k] + 0.5 * dt * k2, **kwargs)
    k4 = F(t[k] + dt, U[k] + dt * k3, **kwargs)
    U[k + 1] = U[k] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

  return U   

# Euler method for testing...
def Euler(t, u0, **kwargs):
  L = len(t)
  U = np.zeros((L, len(u0)))
  U[0] = u0
  dt = t[1] - t[0]
  for k in range(L - 1):
    U[k + 1] = U[k] + dt * F(t[k], U[k], **kwargs)
    
  return U    

# Plot x(t)
def plot(t, x):
  plt.plot(t, x)
  plt.grid(True)
  plt.show()

# Plot solution
def plot3D(x, y, z):
  fig = plt.figure(figsize=(12, 12))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(x, y, z)
  #ax.view_init(elev=90, azim=270)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()
#%% Reproducing Rossler Paper Experiment
L_r = 500000 # 340000
T_max_r = 813.312#339.249
#L_r = int(T_max_r / 0.001)
t_r = np.linspace(0, T_max_r, L_r + 1)
a_r, b_r, c_r = 0.2, 0.2, 5.7
u0_r = np.array([0, -6.78, 0.02])
Ur = RK4(t_r, u0_r, a=a_r, b=b_r, c=c_r)
#Ue = Euler(t, u0, a=a_, b=b_, c=c_)
#Uivp = solve_ivp(fun=lambda t, y: F(t, y, a=a_, b=b_, c=c_), t_span=[t[0], t[-1]], y0=u0, t_eval=t)
#%%
#%matplotlib qt
#U2 = Uivp.y.T
plot3D(Ur[:,0], Ur[:,1], Ur[:,2])
print("Rossler experiment - (x, y, z) at t_end: ", Ur[-1])
#plot(Ue[:,0], Ue[:,1], Ue[:,2])
#plot(U2[:,0], U2[:,1], U2[:,2])
#%%

#print(Ue[-1], Ue[0])
#print(U2[-1], U2[0])

#%% P4
L_1 = 5000 # 5000
T_max_1 = 300
#L = T_max / 0.001
t_1 = np.linspace(0, T_max_1, L_1 + 1)
u0_1 = np.array([4, 0, 0])

#%% Experiment 1
a_, b_, c_1 = 0.1, 0.1, 4
U1 = RK4(t_1, u0_1, a=a_, b=b_, c=c_1)

#%% Experiment 2
c_2 = 6
U2 = RK4(t_1, u0_1, a=a_, b=b_, c=c_2)

#%% Experiment 3
c_3 = 8.5
U3 = RK4(t_1, u0_1, a=a_, b=b_, c=c_3)

#%% Experiment 4
c_4 = 11
U4 = RK4(t_1, u0_1, a=a_, b=b_, c=c_4)

#%% Experiment 5
c_5 = 12.5
U5 = RK4(t_1, u0_1, a=a_, b=b_, c=c_5)

#%% Experiment 6
c_6 = 14
U6 = RK4(t_1, u0_1, a=a_, b=b_, c=c_6)

#%% Plots question 4
plot3D(U1[:,0], U1[:,1], U1[:,2])
plot3D(U2[:,0], U2[:,1], U2[:,2])
plot3D(U3[:,0], U3[:,1], U3[:,2])
plot3D(U4[:,0], U4[:,1], U4[:,2])
plot3D(U5[:,0], U5[:,1], U5[:,2])
plot3D(U6[:,0], U6[:,1], U6[:,2])

#%% P5
L_2 = 5000
T_max_2 = 300
t_2 = np.linspace(0, T_max_1, L_1 + 1)
a_, b_, c_ = 0.2, 0.2, 5.7
u0_2a = np.array([1, 0, 0])
u0_2b = np.array([1.01, 0, 0])

#%%
U7a = RK4(t_2, u0_2a, a=a_, b=b_, c=c_)
U7b = RK4(t_2, u0_2b, a=a_, b=b_, c=c_)

#%%
plot3D(U7a[:,0], U7a[:,1], U7a[:,2])
plot3D(U7b[:,0], U7b[:,1], U7b[:,2])

#%%
plot(t_2, U7a[:,0])
plot(t_2, U7b[:,0])
plot(t_2, U7a[:,0] - U7b[:,0])

#%% Data structure to save
X1 = np.zeros((len(U7a), 2)); X1[:,0] = t_2; X1[:,1] = U7a[:,0]
X2 = np.zeros((len(U7b), 2)); X2[:,0] = t_2; X2[:,1] = U7b[:,0]
X1X2 = np.zeros((len(U7b), 2)); X1X2[:,0] = t_2; X1X2[:,1] = U7a[:,0] - U7b[:,0]

#%% Save data
DIR = 'data/2/'
np.savetxt(DIR + 'Ur.csv', Ur, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U1.csv', U1, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U2.csv', U2, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U3.csv', U3, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U4.csv', U4, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U5.csv', U5, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U6.csv', U6, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U7a.csv', U7a, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'U7b.csv', U7b, fmt='%.8f', delimiter=',', header='x,y,z', comments="")
np.savetxt(DIR + 'X1.csv', X1, fmt='%.8f', delimiter=',', header='t,x', comments="")
np.savetxt(DIR + 'X2.csv', X2, fmt='%.8f', delimiter=',', header='t,x', comments="")
np.savetxt(DIR + 'X1X2.csv', X1X2, fmt='%.8f', delimiter=',', header='t,x', comments="")

