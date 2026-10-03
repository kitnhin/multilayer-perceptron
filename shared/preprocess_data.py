import statistics as st

def readfile(filename):
	file = open(filename)
	file_contents = file.read()
	file.close()
	return file_contents

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

#read and normalise train and validation data
def load_train_val_data(train_file, validation_file):
	train_results, train_data = extract_data(readfile(train_file))
	means, stds = normalise_data(train_data)

	validation_results, validation_data = extract_data(readfile(validation_file))
	normalise_validation_data(validation_data, means, stds) #use training means and stds so both are scaled the same way

	return train_data, train_results, validation_data, validation_results, means, stds