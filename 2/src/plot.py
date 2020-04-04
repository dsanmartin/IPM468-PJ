import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot1D(x, y):
  plt.plot(x, y)
  plt.grid(True)
  #plt.xlim([0, np.max(x)])
  #plt.ylim([0, np.max(y)])
  plt.show()
  
def plot2D(x, y, z):
  plt.contourf(x, y, z)
  #plt.imshow(z, vmin=np.min(z), vmax=np.max(z), 
  #           origin="lower", extent=[0, 1, 0, 1])
  plt.grid(True)
  plt.colorbar()
  plt.show()
  
def plot3D(x, y, z):
  ax = plt.gca(projection='3d')
  ax.plot_surface(x, y, z)
  #ax.set_zlim([0, np.max(z)+ 10])
  plt.show()
  
def quiver(x, y, u, v):
  plt.figure(figsize=(8, 8))
  plt.quiver(x, y, u, v)
  plt.show()
  
def compare(x, y1, y2, y3):
  plt.plot(x, y1, label="Solution")
  plt.plot(x, y2, label="Lax-Friedrichs")
  plt.plot(x, y3, label="Rusanov")
  plt.legend()
  plt.grid(True)
  plt.show()