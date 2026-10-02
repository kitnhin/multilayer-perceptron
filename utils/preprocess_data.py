import argparse
import json
import numpy as np
import statistics as st

def readfile(filename):
	file = open(filename)
	file_contents = file.read()
	file.close()
	return file_contents

def train_parse_args():

	#python argument parser very nice(type defaults to string)
	parser = argparse.ArgumentParser()
	parser.add_argument('--trainFile', default="datasets/dataset_train.csv")
	parser.add_argument('--validationFile', default="datasets/dataset_predict.csv")
	parser.add_argument('--outputFile', default="params.json")
	parser.add_argument('--layer', type=int, nargs='*', default=[24, 24, 24])
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--learningRate', type=float, default=0.1)
	parser.add_argument('--batchSize', type=int, default=30)
	parser.add_argument('--activationFt', default="sigmoid")
	parser.add_argument('--weightsInitialiser', default="heUniform")
	parser.add_argument('--seed', type=int, default=-1)

	return parser.parse_args()

def predict_parse_args():

	#python argument parser very nice(type defaults to string)
	parser = argparse.ArgumentParser()
	parser.add_argument('--predictFile', default="datasets/dataset_predict.csv")
	parser.add_argument('--paramsFile', default="params.json")
	parser.add_argument('--outputFile', default="predictions_output.txt")
	parser.add_argument('--activationFt', default="sigmoid")

	return parser.parse_args()


def write_params(params, filename):
	try:
		f = open(filename, "w")
		f.write(json.dumps(params))
		f.close()
	except Exception:
		print("Failed to write data")

def extract_data(contents):
	lines = contents.strip().split("\n")
	actual_results = []
	data = [] #[x][y] = line x feature y in the given data
	
	for line in lines:
		line_parts = line.strip().split(",")
		#check line parts if want, imma skip this for now
		actual_results.append(line_parts[1])
		line_data = []
		for i in range(2, len(line_parts)):
			line_data.append(float(line_parts[i].strip()))
		data.append(line_data)
	return actual_results, data

def normalise_data(data):
	means = []
	stds = []

	#calculate mean and std for each iteration
	for i in range(len(data[0])): #loop for feature
		feature = []
		for j in range(len(data)): #loop for each line, to get the scores of all lines for each feature
			feature.append(data[j][i])
		means.append(st.mean(feature))
		stds.append(st.stdev(feature))
	
	#normalise each score
	for i in range(len(data)):
		for j in range(len(data[0])):
			data[i][j] = (data[i][j] - means[j]) / stds[j]
	
	return means, stds


def normalise_validation_data(validation_data, training_means, training_stds):
	
	#normalise each score
	for i in range(len(validation_data)):
		for j in range(len(validation_data[0])):
			validation_data[i][j] = (validation_data[i][j] - training_means[j]) / training_stds[j]