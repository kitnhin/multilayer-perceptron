import argparse
import json
import numpy as np
import statistics as st
from shared.other_utils import predict_parse_args
from shared.predict_utils import load_test_data, report_results
import matplotlib.pyplot as plot

from dense import Dense
from activation import Activation

def reconstruct_network(weights, biases, activation):
	network = []

	#all layers
	for i in range(len(weights)):
		dense_layer = Dense(weights[i].shape[1], weights[i].shape[0], "", 1) #shape[1] = col = number of inputs = input size
		dense_layer.weights = weights[i]
		dense_layer.bias = biases[i]
		network.append(dense_layer)
		network.append(Activation(activation[i]))

	return network

def load_model(filename):

	with open(filename, 'r') as f:
		model_data = json.load(f)

	weights = []
	biases = []

	#extract and convert weights and bias to numpy arr
	for weight in model_data["weights"]:
		weights.append(np.array(weight))

	for bias in model_data["biases"]:
		biases.append(np.array(bias))

	means = model_data["means"]
	stds = model_data["stds"]
	activation = model_data["activation"]

	return weights, biases, means, stds, activation


def binary_crossentropy_error(y_pred, y_true):
	epsilon = 1e-15 #to prevent division by 0
	loss = -np.mean((y_true * np.log(y_pred + epsilon)) + ((1 - y_true) * np.log(1 - y_pred + epsilon))) #average of the two neurons output
	return loss



def predict(network, data):
	predicts_arr = []
	
	for i in range(len(data)):
		x = data[i]
		
		#forward pass only
		output = x.reshape(-1, 1)
		for layer in network:
			output = layer.forward(output)

		#get predicted class
		higher_idx = np.argmax(output) #gets the index wif the higher number
		predicted_class = "B" if higher_idx == 0 else "M"
		predicts_arr.append(predicted_class)
	
	return predicts_arr


if __name__ == "__main__":
	try:
		args = predict_parse_args()

		#load model, then test data normalised wif the training means and stds
		weights, biases, means, stds, activation = load_model(args.paramsFile)
		data, actual_results = load_test_data(args.testFile, means, stds)
		data = np.array(data)

		#predict
		network = reconstruct_network(weights, biases, activation)
		predicts = predict(network, data)
		report_results(predicts, actual_results, args.outputFile)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()