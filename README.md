# Research
* Usman Sattar
* G00345816@gmit.ie
### Purpose:
When Completed, the application should predict a number from 0-9 which the user draws with his or her mouse. It should take in the number which any user draws and print it back stating what that number was. 

### Definition of Machine Learning

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning involves you a developer to train a model. For example, training a model to recognise digits.

![Image description](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. Keras supports both convolutional networks and recurrent networks, as well as combinations of the two. It is designed to enable fast experimentation with deep neural networks. 
Before installing Keras, I install one of its backend engines - In this project, I have used the TensorFlow backend, which is also recommened professionally.

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

### References to Keras
* https://keras.io/

![Image description](https://www.lewuathe.com/assets/img/posts/2019-03-06-annoucements-in-tensorflow-dev-summit-2019/catch.png)
TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications. https://www.tensorflow.org/

Uses of TensorFlow:

1. Voice/Sound Recognition
2. Text Based Applications
3. Image Recognition
4. Time Series
5. Video Detection

#### Image Recognition
For our project we have to convert hand-written digits to images. Using TensorFlow we can then predict what number the user had drawn.


![Image description](https://i2.wp.com/dataaspirant.com/wp-content/uploads/2017/05/Mnist-database-hand-written-digits.png?resize=530%2C297)

### References to Keras
* https://www.exastax.com/deep-learning/top-five-use-cases-of-tensorflow/

![Image description](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Flask_logo.svg/1200px-Flask_logo.svg.png)

Flask

https://www.palletsprojects.com/p/flask/
