import numpy as np
from utils import *

class MLQP:
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        self.num_hidden_layers = num_hidden_layers  # num of hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.parameters = dict()
        self.tmp = dict()
        self.grad = dict()

        # initialize network parameters
        for k in range(1, num_hidden_layers + 2):
            self.parameters[k] = dict()
            if k == 1:
                self.parameters[k]["u"] = np.random.normal(size=(input_dim, hidden_dim))
                self.parameters[k]["v"] = np.random.normal(size=(input_dim, hidden_dim))
                self.parameters[k]["b"] = np.random.normal(size=(hidden_dim, 1))
            elif k == num_hidden_layers + 1:
                self.parameters[k]["u"] = np.random.normal(size=(hidden_dim, output_dim))
                self.parameters[k]["v"] = np.random.normal(size=(hidden_dim, output_dim))
                self.parameters[k]["b"] = np.random.normal(size=(output_dim, 1))
            else:
                self.parameters[k]["u"] = np.random.normal(size=(hidden_dim, hidden_dim))
                self.parameters[k]["v"] = np.random.normal(size=(hidden_dim, hidden_dim))
                self.parameters[k]["b"] = np.random.normal(size=(hidden_dim, 1))

        # temporary stored results
        for k in range(0, num_hidden_layers + 2):
            self.tmp[k] = dict()

    def forward(self, input):
        self.tmp[0]["x"] = input[:, np.newaxis]
        for k in range(1, self.num_hidden_layers + 2):
            self.tmp[k]["n"] = np.dot(self.parameters[k]["u"].T, self.tmp[k - 1]["x"] ** 2) \
                             + np.dot(self.parameters[k]["v"].T, self.tmp[k - 1]["x"]) + self.parameters[k]["b"]
            self.tmp[k]["x"] = sigmoid(self.tmp[k]["n"], False)
        return self.tmp[self.num_hidden_layers + 1]["x"]
    
    def backward(self, y):
        self.grad[self.num_hidden_layers + 1] = y - self.tmp[self.num_hidden_layers + 1]["x"]
        for k in range(self.num_hidden_layers, 0, -1):
            delta_n = sigmoid(self.tmp[k + 1]["x"], True)
            self.grad[k] = 2 * self.tmp[k]["x"] * np.dot(self.parameters[k+1]["u"], delta_n * self.grad[k+1]) + \
                           np.dot(self.parameters[k+1]["v"], delta_n * self.grad[k+1])

    def step(self, lr):
        for k in range(1, self.num_hidden_layers + 2):
            delta_n = sigmoid(self.tmp[k]["x"], True)
            self.parameters[k]["u"] += lr * np.outer(self.tmp[k-1]["x"] ** 2, delta_n * self.grad[k])
            self.parameters[k]["v"] += lr * np.outer(self.tmp[k-1]["x"], delta_n * self.grad[k])
            self.parameters[k]["b"] += delta_n * lr * self.grad[k]