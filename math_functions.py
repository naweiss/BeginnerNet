from math import exp

def sigmoid(x):
    return 1 / (1 + exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))