import argparse
import numpy as np

def readfile(filename):
	file = open(filename)
	file_contents = file.read()
	file.close()
	return file_contents

def parse_args():

	#python argument parser very nice(type defaults to string)
	parser = argparse.ArgumentParser()
	parser.add_argument('--givenFile', default="datasets/data-2.csv")
	parser.add_argument('--trainFile', default="datasets/dataset_train.csv")
	parser.add_argument('--validationFile', default="datasets/dataset_validation.csv")
	parser.add_argument('--testFile', default="datasets/dataset_test.csv")
	parser.add_argument('--trainPercentage', type=float, default=0.7)
	parser.add_argument('--validationPercentage', type=float, default=0.15) #predict file gets the rest
	parser.add_argument('--seed', type=int, default=-1)

	return parser.parse_args()

def write_data(data, filename):
	try:
		f = open(filename, "w")
		f.write("\n".join(data)) #join 2d array to a string, as write only works for strings
		f.close()
	except Exception:
		print("Failed to write data")

def reorder_lines(lines, seed):
	
	if seed != -1:
		np.random.seed(seed)

	new_data = []
	random_indexes = np.random.permutation(len(lines))

	for i in random_indexes:
		new_data.append(lines[i])
	
	return new_data

if __name__ == "__main__":
	try:
		#parsing args
		args = parse_args()
		given_file = args.givenFile
		train_percentage = args.trainPercentage
		validation_percentage = args.validationPercentage
		train_file = args.trainFile
		validation_file = args.validationFile
		test_file = args.testFile
		seed = args.seed

		if train_percentage + validation_percentage >= 1:
			raise ValueError("trainPercentage + validationPercentage must be less than 1, need leftover lines for test file")

		#read file
		given_file_contents = readfile(given_file)

		#calculate lines
		lines = given_file_contents.strip().split("\n")
		number_of_lines = len(lines)
		train_lines = int(number_of_lines * train_percentage)
		validation_lines = int(number_of_lines * validation_percentage)

		#calculate where split ends
		train_end = train_lines
		validation_end = train_lines + validation_lines

		#processings
		lines = reorder_lines(lines, seed)

		#write
		write_data(lines[:train_end], train_file)
		write_data(lines[train_end:validation_end], validation_file)
		write_data(lines[validation_end:], test_file) #test file gets the rest
		print(f"Training data saved: {train_file} ({train_end} lines)")
		print(f"Validation data saved: {validation_file} ({validation_end - train_end} lines)")
		print(f"Test data saved: {test_file} ({number_of_lines - validation_end} lines)")
		

	except Exception as e:
		print("Error: ", e)