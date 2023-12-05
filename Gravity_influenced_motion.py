import numpy as np

G = 6.674 * 10**-11
def gravitational_force(m ,n, r):
  F = (G*m*n)/(r**2)
  return F
  m = 5
  n = 3
  r = 2
  force = gravitational_force(m, n, r)

# Print the result
print("Gravitational Force:", force)
