import argparse
import json
import numpy as np
import statistics as st
import preprocess_data as pd
import matplotlib.pyplot as plot

from dense import Dense
from activation import Activation

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
	
	#actual processing
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


def train(network, train_data, train_results, validation_data, validation_results, epochs, learning_rate, batch_size):

	#track loss / error after each epoch to plot later
	loss_arr = [] 
	val_loss_arr = []
	train_acc_arr = []
	val_acc_arr = []

	for epoch in range(epochs):
		error_arr = [] #track error after each iteration

		#use stochastic gradient descend for now cuz its easier
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

		#calc acc
		train_acc = calc_accuracy(network, train_data, train_results)
		val_acc = calc_accuracy(network, validation_data, validation_results)
		train_acc_arr.append(train_acc)
		val_acc_arr.append(val_acc)

		#calc losses for validation data
		validation_loss = calc_validation_loss(network, validation_data, validation_results)
		val_loss_arr.append(validation_loss)
		
		loss = np.mean(error_arr)
		loss_arr.append(loss)
		print(f"{epoch + 1}/{epochs} - loss: {loss:.4f} - val_loss: {validation_loss:.4f}")
	
	return loss_arr, val_loss_arr, train_acc_arr, val_acc_arr


def plot_loss(train_loss_arr, val_loss_arr):
	epochs = range(1, len(train_loss_arr) + 1)
	plot.plot(epochs, train_loss_arr, label='Training Loss', color='blue')
	plot.plot(epochs, val_loss_arr, label='Validation Loss', color='orange')
	plot.xlabel('Epoch')
	plot.ylabel('Loss')
	plot.title('Training and Validation Loss')
	plot.legend()
	plot.show()

def plot_acc(train_acc_arr, val_acc_arr):
	epochs = range(1, len(train_acc_arr) + 1)
	plot.plot(epochs, train_acc_arr, label='Training Accuracy', color='blue')
	plot.plot(epochs, val_acc_arr, label='Validation Accuracy', color='orange')
	plot.xlabel('Epoch')
	plot.ylabel('Accuracy')
	plot.title('Training and Validation Accuracy')
	plot.legend()
	plot.show()

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
		args = pd.train_parse_args()
		train_file = args.trainFile
		output_file = args.outputFile
		layers = args.layer #2d array smth like [24,24]
		epochs = args.epochs
		learning_rate = args.learningRate
		validation_file = args.validationFile
		activation_ft = args.activationFt
		batch_size = args.batchSize
		weights_init = args.weightsInitialiser
		seed = args.seed

		#extract and process training data
		given_file_contents = pd.readfile(train_file)
		actual_results, data = pd.extract_data(given_file_contents)
		data = np.array(data)  #convert list to numpy array
		training_means, training_stds = pd.normalise_data(data)

		#extract and process validation data
		validation_file_contents = pd.readfile(validation_file)
		validation_actual_results, validation_data = pd.extract_data(validation_file_contents)
		validation_data = np.array(validation_data)  #convert list to numpy array
		pd.normalise_validation_data(validation_data, training_means, training_stds) #use training means and stds to ensure acc since our training normalising uses these

		#training
		network = construct_network(data, layers, activation_ft, weights_init, seed)
		loss_arr, val_loss_arr, train_acc_arr, val_acc_arr = train(network, data, actual_results, validation_data, validation_actual_results, epochs, learning_rate, batch_size)
		plot_loss(loss_arr, val_loss_arr)
		plot_acc(train_acc_arr, val_acc_arr)

		save_model(network, training_means, training_stds, output_file)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()