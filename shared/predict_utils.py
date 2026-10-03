import shared.preprocess_data as pd

#read and normalise test data using the means and stds saved from training
def load_test_data(test_file, means, stds):
	actual_results, data = pd.extract_data(pd.readfile(test_file))
	pd.normalise_validation_data(data, means, stds)
	return data, actual_results

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

#print accuracy and write predictions to file
def report_results(predicts, actual, filename):
	accuracy = calc_accuracy(predicts, actual)
	print("Prediction stats: ")
	print(f"Final accuracy: {accuracy:.4f}")
	write_predictions(predicts, actual, filename)
