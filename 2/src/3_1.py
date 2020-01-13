import numpy as np
import matplotlib.pyplot as plt
#%%

h0 = 5
g = 9.8

c0 = np.sqrt(g * h0)

xA = lambda t: -c0 * t
xB = lambda t: 2 * c0 * t

def h_(x, t):
  o = np.zeros_like(x)
  o[x <= xA(t)] = h0
  # Second condition
  #idx = np.logical_and(x >= xA(t), x <= xB(t))[0]
  for i in range(len(x)):
    if xA(t) <= x[i] <= xB(t):
      print(i)
      o[i] = (-x[i]/t + 2 * c0) ** 2 / (9 * g)
  
  return o
#  o = np.zeros_like(x)
#  for i in range(len(x)):
#    if x[i] <= xA(t):
#      o[i] = h0
#    elif xA(t) <= x[i] <= xB(t):
#      o[i] = (-x[i]/t + 2 * c0) ** 2 / (9 * g)
#    elif x[i] >= xB(t):
#      o[i] = 0
#  return o
  
def u_(x, t):
  if x <= xA(t):
    return 0
  elif xA(t) <= x <= xB(t):
    return 2 / 3 * (x / t + c0)
  elif x >= xB(t):
    return 0
  
h = lambda x, t: h_(x, t)
#%%
x = np.linspace(-10, 10, 500)

plt.plot(x, h(x, 1))
plt.show()