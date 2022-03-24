import numpy as np
from utils import *

class MLQP:
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        self.num_layers = num_layers  # num of hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.parameters = dict()
        self.tmp = dict()
        self.grad = dict()

        # initialize network parameters
        for k in range(1, num_layers + 2):
            self.parameters[k] = dict()
            if k == 1:
                self.parameters[k]["u"] = np.random.normal((input_dim, hidden_dim))
                self.parameters[k]["v"] = np.random.normal((input_dim, hidden_dim))
                self.parameters[k]["b"] = np.random.normal((hidden_dim, 1))
            elif k == num_layers + 1:
                self.parameters[k]["u"] = np.random.normal((hidden_dim, output_dim))
                self.parameters[k]["v"] = np.random.normal((hidden_dim, output_dim))
                self.parameters[k]["b"] = np.random.normal((output_dim, 1))
            else:
                self.parameters[k]["u"] = np.random.normal((hidden_dim, hidden_dim))
                self.parameters[k]["v"] = np.random.normal((hidden_dim, hidden_dim))
                self.parameters[k]["b"] = np.random.normal((hidden_dim, 1))

        # temporary stored results
        for k in range(0, num_layers + 2):
            self.tmp[k] = dict()

    def forward(self, input):
        self.tmp[0]["x"] = input[:, np.newaxis]
        for k in range(1, self.num_layers + 2):
            self.tmp[k]["n"] = self.parameters[k]["u"].dot(self.tmp[k - 1]["x"]*self.tmp[k - 1]["x"]) \
                             + self.parameters[k]["v"].dot(self.tmp[k - 1]["x"]) + self.parameters[k]["b"]
            self.tmp[k]["x"] = sigmoid(self.tmp[k]["n"], False)
    
    def backward(self, y):
        delta_x = y - self.tmp[self.num_layers + 1]["x"]
        delta_n = sigmoid(self.tmp[self.num_layers + 1]["x"], True)
        self.grad[self.num_layers + 1] = delta_x * delta_n
        for k in range(self.num_layers, -1, -1):
            self.grad[k] = sigmoid(self.tmp[k]["x"], True) * \
                           ((2 * self.tmp[k]["x"]) * (self.parameters[k+1]["u"].T.dot(self.grad[k+1])) + \
                           self.parameters[k+1]["v"].T.dot(self.grad[k+1]))

    def step(self, lr):
        for k in range(1, self.num_layers + 2):
            self.parameters[k]["u"] -= lr * np.outer(self.grad[k], self.tmp[k-1]["x"] * self.tmp[k-1]["x"])
            self.parameters[k]["v"] -= lr * np.outer(self.grad[k], self.tmp[k-1]["x"])
            self.parameters[k]["b"] -= lr * self.grad[k]