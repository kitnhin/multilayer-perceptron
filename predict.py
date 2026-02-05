import argparse
import json
import numpy as np
import statistics as st
import preprocess_data as pd
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


def predict(network, data, results):
	predicts_arr = []
	error_arr = []
	
	for i in range(len(data)):
		x = data[i]
		correct_result = np.array([[1],[0]]) if results[i] == "B" else np.array([[0],[1]])
		
		#forward pass only
		output = x.reshape(-1, 1)
		for layer in network:
			output = layer.forward(output)

		#get predicted class
		higher_idx = np.argmax(output) #gets the index wif the higher number
		predicted_class = "B" if higher_idx == 0 else "M"
		predicts_arr.append(predicted_class)

		error = binary_crossentropy_error(output, correct_result)
		error_arr.append(error)
	
	return np.mean(error_arr), predicts_arr

def calc_accuracy(predicts, actual):
	correct = 0
	total = len(actual)

	for i in range(len(predicts)):
		if predicts[i] == actual[i]:
			correct += 1

	accuracy = correct / total
	return accuracy

def write_predictions(predicts, actual, filename):
	try:
		with open(filename, "w") as f:
			#write header
			f.write("| actual | predict | results  |\n")
			f.write("|--------|---------|----------|\n")
			
			#write predictions
			for i in range(len(predicts)):
				result = "   ✅   " if actual[i] == predicts[i] else "   ❌   "
				f.write(f"|   {actual[i]}    |    {predicts[i]}    | {result} |\n")
		
		print(f"Results written to {filename}")
	except Exception as e:
		print(f"Failed to write data: {e}")


if __name__ == "__main__":
	try:
		#parsing args
		args = pd.predict_parse_args()
		predict_file = args.predictFile
		params_file = args.paramsFile
		output_file = args.outputFile

		#extract and process training data
		given_file_contents = pd.readfile(predict_file)
		actual_results, data = pd.extract_data(given_file_contents)
		data = np.array(data)  #convert list to numpy array

		#extract params
		weights, biases, means, stds, activation = load_model(params_file)

		#process
		pd.normalise_validation_data(data, means, stds)
		network = reconstruct_network(weights, biases, activation)
		avg_error, predicts = predict(network, data, actual_results)
		accuracy = calc_accuracy(predicts, actual_results)
		
		#output
		print("Prediction stats: ")
		print(f"Average error: {avg_error:.4f}")
		print(f"Final accuracy: {accuracy:.4f}")
		write_predictions(predicts, actual_results, output_file)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()