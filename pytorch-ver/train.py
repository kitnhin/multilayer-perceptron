import torch
import torch.nn as nn # nn for neural network
import utils.preprocess_data as pd
from utils.plot import plot_loss, plot_acc

def construct_network(data, layers, activation_ft, weights_init, seed):
	if seed != -1:
		torch.manual_seed(seed)

	# pick activation function
	if activation_ft == "relu":
		Activation = nn.ReLU
	else:
		Activation = nn.Sigmoid

	#construct network
	network = []

	print("=================================================================================\n")

	#first layer
	input_size = len(data[0]) #number of fields
	output_size = layers[0]
	network.append(nn.Linear(input_size, output_size)) #first input layer
	network.append(Activation())
	prev_out_size = output_size
	print (f"input layer (neurons: {output_size}) -> ", end="")

	#hidden layers
	for i in range(len(layers)):
		input_size = prev_out_size
		output_size = layers[i]
		network.append(nn.Linear(input_size, output_size))
		network.append(Activation())
		prev_out_size = output_size
		print (f"hidden layer {i + 1} (neurons: {output_size}) -> ", end="")

	#output layer
	input_size = prev_out_size
	output_size = 2 #follow picture in pdf, they used 2 neurons in last layer
	network.append(nn.Linear(input_size, output_size)) # No need softmax here, cross entropy loss auto uses it for pytorch
	print (f"output layer (neurons: {output_size})\n")
	print("=================================================================================\n")

	#initialize weights
	for layer in network:
		if isinstance(layer, nn.Linear):
			if weights_init == "heUniform":
				nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
			else:
				nn.init.normal_(layer.weight)
			nn.init.zeros_(layer.bias)

	return nn.Sequential(*network)

def calc_accuracy(network, data, results):
	correct = 0
	total = len(data)
	for i in range(len(data)):
		x = data[i]
		correct_result = results[i].item() #tensor to python int (0 = B, 1 = M)

		#forward pass only so no need store input to calc gradient for backprop 
		with torch.no_grad():
			output = network(x)

		higher_idx = output.argmax().item() #gets the index wif the higher number

		if higher_idx == correct_result:
			correct += 1
		
	accuracy = correct / total
	return accuracy

def calc_validation_loss(network, data, results, loss_ft):
	error_arr = []
	
	for i in range(len(data)):
		x = data[i]
		correct_result = results[i]
		
		#forward pass only
		with torch.no_grad():
			output = network(x)
		
		error = loss_ft(output, correct_result)
		error_arr.append(error.item())
	
	return sum(error_arr) / len(error_arr)

def save_model(network, means, stds, layers, activation_ft, filename):
	torch.save({ #json cant store tensors, so need torch.save 
		"weights": network.state_dict(),
		"means": means,
		"stds": stds,
		"layers": layers,
		"activation_ft": activation_ft,
	}, filename)
	print(f"Model successfully saved to {filename}")


def train(network, train_data, train_results, validation_data, validation_results, epochs, learning_rate, batch_size):
	#track loss / error after each epoch to plot later
	train_loss_arr = [] 
	val_loss_arr = []
	train_acc_arr = []
	val_acc_arr = []

	loss_ft = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate) #optimizer is used to update weights once the gradients are known (SGD = stochastic gradient descent)
	# there are more optimizers like Adam, but SGD is used for the numpy ver so just use this for now

	for epoch in range(epochs):
		error_arr = [] #track error after each iteration

		#use stochastic gradient descend for now cuz its easier
		for i in range(len(train_data)):
			x = train_data[i]
			correct_result = train_results[i]

			#forward part
			output = network(x)
			
			#calculate loss
			error = loss_ft(output, correct_result)
			error_arr.append(error.item())
			
			#backward pass
			error = error / batch_size #average the loss for the batch
			error.backward() #calculate gradients for each layer

			# update gradient if reach batch size
			if (i + 1) % batch_size == 0 or (i + 1) == len(train_data):
				optimizer.step() #update weights
				optimizer.zero_grad() #reset gradients to 0 for next iteration

		#calc acc and validation loss
		train_loss = sum(error_arr) / len(error_arr)
		train_acc = calc_accuracy(network, train_data, train_results)
		val_acc = calc_accuracy(network, validation_data, validation_results)
		validation_loss = calc_validation_loss(network, validation_data, validation_results, loss_ft)

		train_loss_arr.append(train_loss)
		train_acc_arr.append(train_acc)
		val_loss_arr.append(validation_loss)
		val_acc_arr.append(val_acc)
		
		print(f"{epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_loss: {validation_loss:.4f}")
	
	return train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr

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
		training_file_contents = pd.readfile(train_file)
		training_actual_results, training_data = pd.extract_data(training_file_contents)
		training_means, training_stds = pd.normalise_data(training_data)
		training_data = torch.tensor(training_data, dtype=torch.float32) #convert to tensor, pytorch store their weights in float32
		training_actual_results = torch.tensor([0 if result == "B" else 1 for result in training_actual_results]) #B = 0, M = 1

		#extract and process validation data
		validation_file_contents = pd.readfile(validation_file)
		validation_actual_results, validation_data = pd.extract_data(validation_file_contents)
		pd.normalise_validation_data(validation_data, training_means, training_stds) #use training means and stds to ensure acc since our training normalising uses these
		validation_data = torch.tensor(validation_data, dtype=torch.float32)
		validation_actual_results = torch.tensor([0 if result == "B" else 1 for result in validation_actual_results])

		#training
		network = construct_network(training_data, layers, activation_ft, weights_init, seed)
		loss_arr, val_loss_arr, train_acc_arr, val_acc_arr = train(network, training_data, training_actual_results, validation_data, validation_actual_results, epochs, learning_rate, batch_size)
		plot_loss(loss_arr, val_loss_arr)
		plot_acc(train_acc_arr, val_acc_arr)

		save_model(network, training_means, training_stds, layers, activation_ft, output_file)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()