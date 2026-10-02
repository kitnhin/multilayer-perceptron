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
	parser.add_argument('--predictFile', default="datasets/dataset_predict.csv")
	parser.add_argument('--trainPercentage', type=float, default=0.7)
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
		train_file = args.trainFile
		predict_file = args.predictFile
		seed = args.seed

		#read file
		given_file_contents = readfile(given_file)
		
		#calculate lines
		lines = given_file_contents.strip().split("\n")
		number_of_lines = len(lines)
		train_lines = int(number_of_lines * train_percentage)

		#processings
		lines = reorder_lines(lines, seed)

		#write
		write_data(lines[:train_lines], train_file)
		write_data(lines[train_lines:], predict_file)
		print(f"Training data saved: {train_file}")
		print(f"Prediction data saved: {predict_file}")
		

	except Exception as e:
		print("Error: ", e)