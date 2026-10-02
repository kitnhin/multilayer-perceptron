import torch
import torch.nn as nn
import utils.preprocess_data as pd
from train import construct_network

def load_model(filename):
	saved = torch.load(filename)
	return saved["weights"], saved["means"], saved["stds"], saved["layers"], saved["activation_ft"]

def reconstruct_network(data, weights, layers, activation_ft):
	network = construct_network(data, layers, activation_ft, "random", -1)
	network.load_state_dict(weights)
	return network

def predict(network, data):
	predicts_arr = []
	
	for i in range(len(data)):
		x = data[i]
		
		#forward pass only
		with torch.no_grad():
			output = network(x)

		#get predicted class
		higher_idx = output.argmax().item()
		predicted_class = "B" if higher_idx == 0 else "M"
		predicts_arr.append(predicted_class)
	
	return predicts_arr

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
		test_file = args.testFile
		params_file = args.paramsFile
		output_file = args.outputFile

		#extract params
		weights, means, stds, layers, activation = load_model(params_file)
		
		#extract and process training data
		given_file_contents = pd.readfile(test_file)
		actual_results, data = pd.extract_data(given_file_contents)
		pd.normalise_validation_data(data, means, stds)
		data = torch.tensor(data, dtype=torch.float32) 
		
		#process
		network = reconstruct_network(data, weights, layers, activation)
		predicts = predict(network, data)
		accuracy = calc_accuracy(predicts, actual_results)
		
		#output
		print("Prediction stats: ")
		print(f"Final accuracy: {accuracy:.4f}")
		write_predictions(predicts, actual_results, output_file)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()