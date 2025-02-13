import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline

rng = np.random.default_rng()
x= np.linspace(-3, 3, 50)
y = np.exp(-x) + rng.standard_normal(50) * 0.03

t = [-1,0,1]
k = 3
t = np.r_[(x[0],)*(k+1),t,(x[-1],)*(k+1)]
spl = make_lsq_spline(x, y, t, k)

print(x)
print(y)
print(t)

xs = np.linspace(-3, 3, 100)

plt.plot(x, y, 'o')
plt.plot(xs, spl(xs), '-')
plt.show()