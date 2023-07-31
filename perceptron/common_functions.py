import numpy as np


# step activation function
def step(weighted_sum):
    return 0 if weighted_sum <= 0 else 1


# sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
