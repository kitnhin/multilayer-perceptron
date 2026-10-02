import numpy as np
from layer import Layer

class Dense(Layer):
	def __init__(self, input_size, output_size, initialiser, seed=None):

		if seed != -1:
			np.random.seed(seed)
		
		if initialiser == "heUniform":
			limit = np.sqrt(6 / input_size)
			self.weights = np.random.uniform(-limit, limit, size=(output_size, input_size))
		else:
			self.weights = np.random.randn(output_size, input_size) #in case randn give too big, might change ltr (row = outputsize, col = inputsize)

		self.bias = np.zeros((output_size, 1))

		# gradient accumulation for batch 
		self.weights_gradient_sum = np.zeros_like(self.weights) #zeros like gives an array wif the same size as the one given but all 0s inside
		self.bias_gradient_sum = np.zeros_like(self.bias)
		self.batch_size = 0

	
	def forward(self, input):
		self.input = input
		return np.dot(self.weights, input) + self.bias #outputs y
	
	def backward(self, output_gradient, learning_rate):
		#calculate gradient
		weights_gradient = np.dot(output_gradient, self.input.transpose())
		bias_gradient = output_gradient
		input_gradient = np.dot(self.weights.transpose(), output_gradient)

		#calc gradient sum for batch
		self.weights_gradient_sum += weights_gradient
		self.bias_gradient_sum += bias_gradient
		self.batch_size += 1

		return input_gradient #return error gradient of previous layer
	
	def update_weights(self, learning_rate):
		self.weights -= learning_rate * (self.weights_gradient_sum / self.batch_size)
		self.bias -= learning_rate * (self.bias_gradient_sum / self.batch_size)

		#reset sum arrays
		self.weights_gradient_sum = np.zeros_like(self.weights)
		self.bias_gradient_sum = np.zeros_like(self.bias)
		self.batch_size = 0

	
#variables inside each dense layer
# weights - 2D array of [number of neuron / number of outuput][number of input]
# bias - 1D array of [numer of neuron / number of output]
# input - 1D array of [number of input]