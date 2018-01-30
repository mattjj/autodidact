from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# We could use np.tanh, but let's write our own as an example.
def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x))

x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, grad(tanh)(x),                                # first  derivative
         x, grad(grad(tanh))(x),                          # second derivative
         x, grad(grad(grad(tanh)))(x),                    # third  derivative
         x, grad(grad(grad(grad(tanh))))(x),              # fourth derivative
         x, grad(grad(grad(grad(grad(tanh)))))(x),        # fifth  derivative
         x, grad(grad(grad(grad(grad(grad(tanh))))))(x))  # sixth  derivative

plt.axis('off')
plt.savefig("tanh.png")
plt.show()
