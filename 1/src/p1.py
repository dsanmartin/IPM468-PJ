"""
  Question 1 prototype (only testing...)
"""
import numpy as np
import matplotlib.pyplot as plt
#%% Functions
# Naive
def f(N):
  A = 0
  for n in range(1, N + 1):
    A += 1 / n
  return A

# Pythonic
fv = lambda n: np.sum(1 / n)
#%% Experiments
EXP = 25
Ns = np.array([2 ** i for i in range(EXP+1)])
#As = np.array([f(Ns[i]) for i in range(len(Ns))])
Avs = np.array([fv(np.arange(1, n+1)) for n in Ns])
plt.plot(Ns, Avs)
plt.grid(True)
plt.show()
print(Avs[-1])
#%% For PGFPlot
for i in range(len(Ns)):
  print("%d %f" % (Ns[i], Avs[i]))
##%%
#N = 2**25
#n = np.arange(1, N + 1)
#A = np.array([fv(np.arange(1, i + )) for i in n])
##%%
#%matplotlib inline
#plt.plot(n, A)
#plt.show()
#print(A[-1])