import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import minimize  # Python version of R's optim() function
from sklearn import datasets

# Carry out the exercises in your own copy of the notebook that you can find at
# https://www.kaggle.com/code/datasniffer/perceptrons-mlp-s-and-gradient-descent
# Then copy and paste code asked for below in between the dashed lines.
# Do not import additional packages.

# Task 1:
# Instructions:
# In the notebook, you wrote a function that implements an MLP with
# 2 hidden layers.
# The function should accept a vector of weights and a matrix X that stores
# input feature vectors in its **columns**.
# The name of the function should be my_mlp.

# Copy and paste the code for that function here:
# -----------------------------------------------


def my_mlp(w: np.ndarray, X: np.ndarray, n_input: int=6,
           n_hidden_1: int=4, n_hidden_2: int=7,
           sigma=np.tanh) -> np.ndarray:
    """
    Implement multilayer perceptron with two hidden layers.

    Args:
        w (np.ndarray): Flattened weight vector.
        X (np.ndarray): Input matrix.
        n_input (int, optional): Number of input nodes. Default is 6.
        n_hidden_2 (int, optional): Number of nodes in the second hidden layer.
                                    Default is 7.
        sigma (Callable[[np.ndarray], np.ndarray], optional):
                        Activation function applied
                        element-wise to hidden layers' outputs.
                        Default is np.tanh.

    Returns:
        np.ndarray: Output vector of shape (1, n_samples).
    """

    # Reshape weight vector for each layer
    W1 = w[0: n_input * n_hidden_1].reshape((n_hidden_1, n_input))
    W2 = w[W1.size: W1.size + n_hidden_1 * n_hidden_2].reshape(
        (n_hidden_2, n_hidden_1))
    W3 = w[W1.size + W2.size: W1.size + W2.size + n_hidden_2 * 1].reshape(
        (1, n_hidden_2))

    # Forward pass
    a1 = sigma(W1 @ X)
    a2 = sigma(W2 @ a1)
    f = sigma(W3 @ a2)

    return f
# -----------------------------------------------

# Task 2:
# Instructions:
# In the notebook, you wrote a function that implements a loss function
# for training the MLP implemented by my_mlp of Task 1.
# The function should accept a vector of weights, a matrix X that stores
# input feature vectors in its **columns**, and a vector y that stores the
# target labels (-1 or +1).
# The name of the function should be MSE_func.

# Copy and paste the code for that function here:
# -----------------------------------------------


def MSE_func(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Sum of Squared Errors (SSE) for a multilayer perceptron.

    Args:
        w (np.ndarray): Weight vector. Structured as required by `my_mlp()`.
        X (np.ndarray): Input matrix.
        y (np.ndarray): Target output vector.

    Returns:
        float: Computed SSE
    """

    # Forward pass
    f = my_mlp(w, X)

    # Calculate SSE
    sse = np.sum((f - y)**2)

    return sse
# -----------------------------------------------

# Task 3:
# Instructions:
# In the notebook, you wrote a function that returns the gradient vector for
# the least squares (simple) linear regression loss function.
# The function should accept a vector beta that contains the intercept (β₀)
# and the slope (β₁), a vector x that stores values of the independent
# variable, and a vector y that stores the values of the dependent variable and
# should return an np.array() that has the derivative values as its components.
# The name of the function should be dR.

# Copy and paste the code for that function here:
# -----------------------------------------------


def dR(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute partial derivatives of a loss function wrt. intercept and slope
    in a simple linear regression model.

    Args:
        beta (np.ndarray): Parameter vector of intercept and slope.
        x (np.ndarray): Input vector representing independent variable.
        y (np.ndarray): Input vector representing dependent variable.

    Returns:
        np.ndarray: Gradient vector of partial derivatives wrt.
                    intercept and slope.
    """

    # Compute partial derivatives wrt. intercept and slope.
    dbeta_0 = 2 * np.mean(beta[0] + beta[1] * x - y)
    dbeta_1 = 2 * np.mean(x * (beta[0] + beta[1] * x - y))

    return np.array([dbeta_0, dbeta_1])
# -----------------------------------------------
