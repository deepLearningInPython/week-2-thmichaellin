import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import minimize  # Python version of R's optim() function
from sklearn import datasets
 
# Carry out the exercises in your own copy of the notebook that you can find at
#    https://www.kaggle.com/code/datasniffer/perceptrons-mlp-s-and-gradient-descent.
# Then copy and paste code asked for below in between the dashed lines.
# Do not import additional packages.
 
# Task 1:
# Instructions:
# In the notebook, you wrote a function that implements an MLP with 2 hidden layers.
# The function should accept a vector of weights and a matrix X that stores input feature
# vectors in its **columns**.
# The name of the function should be my_mlp.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def my_mlp(w, X, sigma=np.tanh):

    return f
# -----------------------------------------------
 
# Task 2:
# Instructions:
# In the notebook, you wrote a function that implements a loss function for training
# the MLP implemented by my_mlp of Task 1.
# The function should accept a vector of weights, a matrix X that stores input feature
# vectors in its **columns**, and a vector y that stores the target labels (-1 or +1).
# The name of the function should be MSE_func.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def MSE_func(w, X, y): # give the appropriate name and arguments

    return MSE
# -----------------------------------------------
 
# Task 3:
# Instructions:
# In the notebook, you wrote a function that returns the gradient vector for the least
# squares (simple) linear regression loss function.
# The function should accept a vector beta that contains the intercept (β₀) and the slope (β₁),
# a vector x that stores values of the independent variable, and a vector y that stores
# the values of the dependent variable and should return an np.array() that has the derivative values
# as its components.
# The name of the function should be dR.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def dR(beta, x, y):
    # implement the above formula for dR/dβ₀
    # implement the above formula for dR/dβ₁
    return np.array([dbeta_0, dbeta_1])
 
# -----------------------------------------------
