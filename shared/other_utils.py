import argparse

def train_parse_args():

	#python argument parser very nice(type defaults to string)
	parser = argparse.ArgumentParser()
	parser.add_argument('--trainFile', default="datasets/dataset_train.csv")
	parser.add_argument('--validationFile', default="datasets/dataset_validation.csv")
	parser.add_argument('--outputFile', default="params.json")
	parser.add_argument('--layer', type=int, nargs='*', default=[24, 24, 24])
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--learningRate', type=float, default=0.1)
	parser.add_argument('--batchSize', type=int, default=30)
	parser.add_argument('--activationFt', default="sigmoid")
	parser.add_argument('--weightsInitialiser', default="heUniform")
	parser.add_argument('--seed', type=int, default=-1)
	parser.add_argument('--patience', type=int, default=-1) #-1 means no early stopping

	return parser.parse_args()

def predict_parse_args():

	#python argument parser very nice(type defaults to string)
	parser = argparse.ArgumentParser()
	parser.add_argument('--testFile', default="datasets/dataset_test.csv")
	parser.add_argument('--paramsFile', default="params.json")
	parser.add_argument('--outputFile', default="predictions_output.txt")
	parser.add_argument('--activationFt', default="sigmoid")

	return parser.parse_args()
