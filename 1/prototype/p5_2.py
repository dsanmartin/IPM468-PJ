import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
def plot2D(x, y, u):
  plt.contourf(x, y, u)
  plt.colorbar()
  plt.show()

def ConjugateGradient(A, b, x, conv=1e-6, max_it=2500):
  r = b - np.dot(A, x)
  p = r
  i = 0
  while np.linalg.norm(r) > conv and i <= max_it:
    a = np.dot(p.T, r) / (np.dot(p.T, np.dot(A, r)))
    x = x + a * p  
    r = r - a * np.dot(A, p)
    b = - np.dot(p.T, np.dot(A, r)) / np.dot(np.dot(A, p).T, p)
    p = r + b * p
    i += 1
    print(np.linalg.norm(r))
    
  return x

def createSystem(Nx, Ny):
  D = np.array([[-4, 1, 0], [1, -4, 1], [0, 1, -4]])
  I = np.eye(3)
  A = np.zeros((3 * (Nx - 1), 3 * (Ny - 1)))
  for i in range(Nx - 1):#int(Nx ** 0.5) + 1):
    for j in range(Ny - 1):#int(Ny ** 0.5) + 1):
      if i == j:
        A[3*i:3*i + 3, 3*j:3*j + 3] = D
      elif abs(i - j) == 1:
        A[3*i:3*i + 3, 3*j:3*j + 3] = I
        
  return A
#%% CG testing
N = 100
A = np.random.rand(N, N) # random matrix
A += np.eye(N) * 5 + A.T # ensure symmetric and positive definite
b = np.random.rand(N)

x = np.linalg.solve(A, b)

x_ = ConjugateGradient(A, b, np.random.rand(len(b)))

print(np.linalg.norm(x - x_))

#%% Analytic solution 
p = lambda x, y: np.cos(x + y) * np.sin(x - y)
f = lambda x, y: -4 * p(x, y)
#%%
h = 0.025
Nx, Ny = int(1/h), int(1/h)

x = np.linspace(0, 1, Nx + 1)
y = np.linspace(0, 1, Ny + 1)

X, Y = np.meshgrid(x, y)
#
#u_N = p(x, 1)
#u_S = p(x, 0)
#u_W = p(0, y)
#u_E = p(1, y)
#
#F = f(X, Y)
#b = h ** 2 * F[1:-1,1:-1].flatten()
#b[:Nx + 1] += u_S
#b[-(Nx + 1):] += u_N
#b[::Nx+1] += u_W
#b[Nx::Nx+1] += u_E
#b[::-(Ny+1)] += u_E[::-1]
#b[::-(Ny+1)] += u_W[::-1]
#
#A = createSystem(Nx, Ny)
#%%
#plot2D(X, Y, p(X, Y))
#%%
A = np.zeros([(Nx+1)*(Ny+1), (Nx+1)*(Ny+1)])
b = np.zeros([(Nx+1)*(Ny+1), 1])
dx = dy = h

# Define global indexing
def index(i, j, nCols=(Ny+1)):
  return j + i*nCols
  
for i in range(Nx+1):
  for j in range(Ny+1):
    k = index(i,j)
    if j==0: # y=ymin
      A[k,k] = 1.
      #b[k] = f(x[i], 0)
    elif i==Nx: # x=xmax
      A[k,k] = 1.
      #b[k] = f(1, y[j])
    elif j==Ny: # y=ymax
      A[k,k] = 1.
      #b[k] = f(x[i], 1)
    elif i==0: # x=xmin
      A[k,k] = 1.
      #b[k] = f(1, y[j])
    else:
      A[k, k] = -2./dx**2 - 2./dy**2
      A[k,index(i+1,j)] = 1./dx**2
      A[k,index(i-1,j)] = 1./dx**2
      A[k,index(i,j-1)] = 1./dy**2
      A[k,index(i,j+1)] = 1./dy**2
    b[k] = f(x[i], y[j])

#%%
#X0 = np.zeros((Nx + 1, Ny + 1))
#X0[-1,:] = p(x, 1)
#X0[0,:] = p(x, 0)
#X0[:,0]= p(0, y)
#X0[:,-1] = p(1, y)

U_a = np.linalg.solve(A, b)
#U_a = ConjugateGradient(A, b, b)# + np.random.rand(len(b)))

#%%
U_a = U_a.reshape(Nx + 1, Ny + 1)
plot2D(X, Y, p(X, Y))
plot2D(X, Y, U_a)
plt.imshow(U_a[1:-1,1:-1])
plt.colorbar()