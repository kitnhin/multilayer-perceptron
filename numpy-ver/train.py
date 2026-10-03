import argparse
import json
import numpy as np
import statistics as st
from shared.other_utils import train_parse_args

from dense import Dense
from activation import Activation
from shared.preprocess_data import load_train_val_data
from shared.training_utils import create_history, record_epoch, plot_history, print_final_accuracy, create_early_stop_state, early_stop_check

def construct_network(data, layers, activation_ft, weights_init, seed):
	#construct network
	network = []

	print("===================================== network =====================================\n")
	#first layer
	input_size = len(data[0]) #number of fields
	output_size = layers[0]
	network.append(Dense(input_size, output_size, weights_init, seed)) #first input layer
	network.append(Activation(activation_ft))
	prev_out_size = output_size
	print (f"input layer (neurons: {output_size}) -> ", end="")

	#hidden layers
	for i in range(len(layers)):
		input_size = prev_out_size
		output_size = layers[i]
		network.append(Dense(input_size, output_size, weights_init, seed))
		network.append(Activation(activation_ft))
		prev_out_size = output_size
		print (f"hidden layer {i + 1} (neurons: {output_size}) -> ", end="")
	
	#output layer
	input_size = prev_out_size
	output_size = 2 #follow picture in pdf, they used 2 neurons in last layer
	network.append(Dense(input_size, output_size, weights_init, seed))
	network.append(Activation("softmax"))
	print (f"output layer (neurons: {output_size})\n")
	print("=================================================================================\n")

	return network

def calc_validation_loss(network, data, results):
	error_arr = []
	
	for i in range(len(data)):
		x = data[i]
		correct_result = np.array([[1],[0]]) if results[i] == "B" else np.array([[0],[1]])
		
		#forward pass only
		output = x.reshape(-1, 1)
		for layer in network:
			output = layer.forward(output)
		
		error = binary_crossentropy_error(output, correct_result)
		error_arr.append(error)
	
	return np.mean(error_arr)

def calc_accuracy(network, data, results):
	correct = 0
	total = len(data)
	for i in range(len(data)):
		x = data[i]
		correct_result = results[i]
		
		#forward pass only
		output = x.reshape(-1, 1)
		for layer in network:
			output = layer.forward(output)
		
		higher_idx = np.argmax(output) #gets the index wif the higher number
		predicted_class = "B" if higher_idx == 0 else "M"

		if predicted_class == correct_result:
			correct += 1
		
	accuracy = correct / total
	return accuracy

def binary_crossentropy_error(y_pred, y_true):
	epsilon = 1e-15 #to prevent division by 0
	loss = -np.mean((y_true * np.log(y_pred + epsilon)) + ((1 - y_true) * np.log(1 - y_pred + epsilon))) #average of the two neurons output
	return loss


def train(network, train_data, train_results, validation_data, validation_results, epochs, learning_rate, batch_size, patience):

	history = create_history()
	early_stop_state = create_early_stop_state()

	for epoch in range(1, epochs + 1):
		error_arr = []

		for i in range(len(train_data)):
			x = train_data[i]
			correct_result = np.array([[1],[0]]) if train_results[i] == "B" else np.array([[0],[1]]) #python op

			#forward part
			output = x.reshape(-1, 1) #need to transpose cuz need convert it to column vector
			for layer in network:
				output = layer.forward(output)
			
			#calculate loss
			error = binary_crossentropy_error(output, correct_result)
			error_arr.append(error)

			#backward
			gradient = output - correct_result

			for layer in reversed(network):
				gradient = layer.backward(gradient, learning_rate)
			
			#update gradient if reach batch size
			if (i + 1) % batch_size == 0 or (i + 1) == len(train_data):
				for layer in network:
					if isinstance(layer, Dense):
						layer.update_weights(learning_rate)

		#calc loss and acc
		train_loss = np.mean(error_arr)
		validation_loss = calc_validation_loss(network, validation_data, validation_results)
		train_acc = calc_accuracy(network, train_data, train_results)
		val_acc = calc_accuracy(network, validation_data, validation_results)
		record_epoch(history, epoch, epochs, train_loss, validation_loss, train_acc, val_acc)

		if early_stop_check(early_stop_state, network, validation_loss, epoch, patience):
			print(f"\nEarly stopping at epoch {epoch}, val_loss has not improved for {patience} epochs")
			break

	#go back to the best network, not the last one
	if early_stop_state["best_network"] is not None:
		network = early_stop_state["best_network"]
		print(f"Restored best weights from epoch {early_stop_state['best_epoch']} (val_loss: {early_stop_state['best_val_loss']:.4f})")

	print_final_accuracy(history, early_stop_state)
	return network, history


def save_model(network, means, stds, filename):
	weights = []
	biases = []
	activation = []
	
	#extract weights and biases
	for layer in network:
		if isinstance(layer, Dense):
			weights.append(layer.weights.tolist())  #convert numpy arr to list
			biases.append(layer.bias.tolist())
			#maybe can add initialiser later
		else:
			activation.append(layer.activation_ft)
	
	model_data = {
		"weights": weights, #weights is an array of 2D arrays
		"biases": biases,
		"means": means,
		"stds": stds,
		"activation": activation
	}

	with open(filename, 'w') as f:
		json.dump(model_data, f)
	
	print(f"Model succesffuly saved to {filename}")


if __name__ == "__main__":
	try:
		#parsing args
		args = train_parse_args()

		#load normalised data, then convert to numpy arrays
		training_data, training_actual_results, validation_data, validation_actual_results, training_means, training_stds = load_train_val_data(args.trainFile, args.validationFile)
		training_data = np.array(training_data)
		validation_data = np.array(validation_data)

		#training
		network = construct_network(training_data, args.layer, args.activationFt, args.weightsInitialiser, args.seed)
		network, history = train(network,training_data, training_actual_results, validation_data, validation_actual_results, args.epochs, args.learningRate, args.batchSize, args.patience)
		plot_history(history)

		save_model(network, training_means, training_stds, args.outputFile)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()