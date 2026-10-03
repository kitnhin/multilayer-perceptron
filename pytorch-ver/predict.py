import torch
import torch.nn as nn
from shared.other_utils import predict_parse_args
from shared.predict_utils import load_test_data, report_results
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

if __name__ == "__main__":
	try:
		args = predict_parse_args()

		#load model, then test data normalised wif the training means and stds
		weights, means, stds, layers, activation = load_model(args.paramsFile)
		data, actual_results = load_test_data(args.testFile, means, stds)
		data = torch.tensor(data, dtype=torch.float32)

		#predict
		network = reconstruct_network(data, weights, layers, activation)
		predicts = predict(network, data)
		report_results(predicts, actual_results, args.outputFile)


	except Exception as e:
		print("Error: ", e)
		import traceback
		traceback.print_exc()