import torch
import torch.nn as nn # nn for neural network
from shared.other_utils import train_parse_args
from shared.preprocess_data import load_train_val_data
from shared.training_utils import create_history, record_epoch, plot_history, print_final_accuracy, create_early_stop_state, early_stop_check

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


def train(network, train_data, train_results, validation_data, validation_results, epochs, learning_rate, batch_size, patience):
	history = create_history()
	early_stop_state = create_early_stop_state()

	loss_ft = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate) #optimizer is used to update weights once the gradients are known (SGD = stochastic gradient descent)
	# there are more optimizers like Adam, but SGD is used for the numpy ver so just use this for now

	for epoch in range(1, epochs + 1): #start from 1 so epoch numbers match what we print
		error_arr = [] #track error after each iteration

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

		#calc loss and acc
		train_loss = sum(error_arr) / len(error_arr)
		validation_loss = calc_validation_loss(network, validation_data, validation_results, loss_ft)
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

if __name__ == "__main__":
	try:
		#parsing args
		args = train_parse_args()

		#load normalised data, then convert to tensors
		training_data, training_actual_results, validation_data, validation_actual_results, training_means, training_stds = load_train_val_data(args.trainFile, args.validationFile)
		training_data = torch.tensor(training_data, dtype=torch.float32) #pytorch store their weights in float32
		training_actual_results = torch.tensor([0 if result == "B" else 1 for result in training_actual_results]) #B = 0, M = 1
		validation_data = torch.tensor(validation_data, dtype=torch.float32)
		validation_actual_results = torch.tensor([0 if result == "B" else 1 for result in validation_actual_results])

		#training
		network = construct_network(training_data, args.layer, args.activationFt, args.weightsInitialiser, args.seed)
		network, history = train(network,training_data, training_actual_results, validation_data, validation_actual_results, args.epochs, args.learningRate, args.batchSize, args.patience)
		plot_history(history)

		save_model(network, training_means, training_stds, args.layer, args.activationFt, args.outputFile)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()