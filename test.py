import matplotlib.pyplot as plt
import numpy as np

xs=np.linspace(0,10,100)

ys=np.sin(xs)

plt.plot(xs,ys)
plt.show()