"""
  Question 1: Partial Harmonic Series
"""
import numpy as np
import matplotlib.pyplot as plt
import pathlib
#%% Functions
# Naive (using loop)
def f(N):
  A = 0
  for n in range(1, N + 1):
    A += 1 / n
  return A

# Vectorized (to get faster results)
fv = lambda n: np.sum(1 / n)

#%% Experiments. Compute some values of series
EXP = 25
Ns = np.array([2 ** i for i in range(EXP+1)]) # N = {2^0, 2^1, ..., 2^EXP}

# Using loop (slow but for N too large may be stored in memory...)
#As = np.array([f(n) for n in Ns], dtype=np.float32)
#Ad = np.array([f(n) for n in Ns], dtype=np.float64)

# Using vectorized implementation (faster but dangerous with memory usage...)
As = np.array([fv(np.arange(1, n+1)) for n in Ns], dtype=np.float32) # Compute summation with single precision
Ad = np.array([fv(np.arange(1, n+1)) for n in Ns], dtype=np.float64) # Compute summation with double precision
diff = np.abs(As-Ad) # Absolute value of difference between single and double precision (a simple measure)

#%% Plot both summations
plt.plot(Ns, As, label="Single precision")
plt.plot(Ns, Ad, label="Double precision")
plt.grid(True)
plt.legend()
plt.show()
#%% Plot difference
plt.plot(Ns, diff, label=r'$|A_d - A_s$|')
plt.grid(True)
plt.yscale('log')
plt.xscale('log', basex=2)
plt.legend()
plt.show()

#%% Data to save
series = np.zeros((len(Ns), 4))
series[:,0] = Ns
series[:,1] = As
series[:,2] = Ad
series[:,3] = diff
#%% Save data for plots
DIR = 'data/1/' # Directory name
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True) # Create Folder
np.savetxt(DIR + 'series.csv', series, fmt='%.16f', delimiter=',', header='N,As,Ad,dif', comments="") # Save data