import numpy as np

def ConjugateGradient(A, b, x, conv=1e-6):
  r = b - np.dot(A, x)
  p = r
  while np.linalg.norm(r) > conv:
    a = np.dot(p.T, r) / (np.dot(p.T, np.dot(A, r)))
    x = x + a * p  
    r = r - a * np.dot(A, p)
    b = - np.dot(p.T, np.dot(A, r)) / np.dot(np.dot(A, p).T, p)
    p = r + b * p
    
  return x
      
      
#%%
N = 100
A = np.random.rand(N, N)
A += np.eye(N) * 10 + A.T
b = np.random.rand(N)
#%%
x = np.linalg.solve(A, b)
print(x)

#%%
x_ = GC(A, b, np.random.rand(len(b)))
print(x_)

#%%
print(np.linalg.norm(x - x_))