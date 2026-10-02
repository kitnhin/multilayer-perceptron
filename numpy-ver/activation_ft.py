import numpy as np

def sigmoid(input):
	return 1 / (1 + np.exp(-input))

def sigmoid_prime(input):
	return np.exp(-input) / (1 + np.exp(-input))**2

def softmax(input): #special activation function that make the output dependent on each other (can read up on this ltr)
	exp_values = np.exp(input - np.max(input))
	return exp_values / np.sum(exp_values)

def relu(input):
	return np.maximum(0, input)

def relu_prime(input):
	return (input > 0).astype(float)