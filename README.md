# Autograd tutorial implementation

A tutorial implementation based on [the full version](https://github.com/hips/autograd).

Example use:

```python
>>> import autograd.numpy as np  # Thinly-wrapped numpy
>>> from autograd import grad    # The only autograd function you may ever need
>>>
>>> def tanh(x):                 # Define a function
...     y = np.exp(-2.0 * x)
...     return (1.0 - y) / (1.0 + y)
...
>>> grad_tanh = grad(tanh)       # Obtain its gradient function
>>> grad_tanh(1.0)               # Evaluate the gradient at x = 1.0
0.41997434161402603
>>> (tanh(1.0001) - tanh(0.9999)) / 0.0002  # Compare to finite differences
0.41997434264973155
```

We can continue to differentiate as many times as we like, and use numpy's
vectorization of scalar-valued functions across many different input values:

```python
>>> import matplotlib.pyplot as plt
>>> x = np.linspace(-7, 7, 200)
>>> plt.plot(x, tanh(x),
...          x, grad(tanh)(x),                                # first  derivative
...          x, grad(grad(tanh))(x),                          # second derivative
...          x, grad(grad(grad(tanh)))(x),                    # third  derivative
...          x, grad(grad(grad(grad(tanh))))(x),              # fourth derivative
...          x, grad(grad(grad(grad(grad(tanh)))))(x),        # fifth  derivative
...          x, grad(grad(grad(grad(grad(grad(tanh))))))(x))  # sixth  derivative
>>> plt.show()
```

Autograd was written by [Dougal Maclaurin](https://dougalmaclaurin.com),
[David Duvenaud](https://www.cs.toronto.edu/~duvenaud/)
and [Matt Johnson](http://people.csail.mit.edu/mattjj/).
See [the main page](https://github.com/hips/autograd) for more information.
