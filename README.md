# Emerging-Tech-Assessment
## Research
### Definition of Machine Learning

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

Machine learning involves you a developer to train a model. For example, training a model to recognise digits.

## Keras
### Neurons in keras 

```python
import keras as kr
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
```
#### Linear
```python
# Create a new neural network.
m = kr.models.Sequential()

# Add a single neuron in a single layer, initialised with weight 1 and bias 0.
m.add(kr.layers.Dense(1, input_dim=1, activation="linear", kernel_initializer='ones', bias_initializer='zeros'))

# Compile the model.
m.compile(loss="mean_squared_error", optimizer="sgd")


# Create some input values.
x = np.arange(0.0, 10.0, 1)

# Run each x value through the neural network.
y = m.predict(x)

x
```

**OUTPUT: array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])**

```python
y
```
**OUTPUT: array([[0.],
       [1.],
       [2.],
       [3.],
       [4.],
       [5.],
       [6.],
       [7.],
       [8.],
       [9.]], dtype=float32)**
     
```python
# Plot the values.
plt.plot(x, y, 'k.')
```
