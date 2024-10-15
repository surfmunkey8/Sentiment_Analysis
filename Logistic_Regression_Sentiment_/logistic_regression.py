import numpy as np

def __init__(self, iterations, learning_rate, theta):

    self.iterations = iterations
    self.learning_rate = learning_rate
    self.theta = theta

    def train_sigmoid(self, X):

        train, validate = train


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def ReLU(x):
    return np.max(0,x)

