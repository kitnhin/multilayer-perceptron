#file paths
TRAIN_DATASET = datasets/dataset_train.csv
PREDICT_DATASET = datasets/dataset_predict.csv
VALIDATION_DATASET = datasets/dataset_predict.csv
GIVEN_DATASET = datasets/data-2.csv

PARAMS_OUTPUT = params.json
PREDICT_OUTPUT = predictions_output.txt

#separation settings
TRAIN_PERCENTAGE = 0.7
SEED = 42 #SEED = -1 means no seed, random

#training configs
EPOCHS = 300
LAYERS = 24 24
LEARNING_RATE = 0.0008
BATCH_SIZE = 1 #put 1 for SGD
ACTIVATION_FT = sigmoid #sigmoid or relu
WEIGHTS_INITIALISER = random #heUniform or random


sep:
	@python3 separate.py --givenFile ${GIVEN_DATASET} --trainFile ${TRAIN_DATASET} --predictFile ${PREDICT_DATASET} --trainPercentage ${TRAIN_PERCENTAGE} --seed ${SEED}

train:
	@python3 train.py --trainFile ${TRAIN_DATASET} --outputFile ${PARAMS_OUTPUT} --layer ${LAYERS} --epochs ${EPOCHS} --learningRate ${LEARNING_RATE} --validationFile ${VALIDATION_DATASET} \
	--batchSize ${BATCH_SIZE} --activationFt ${ACTIVATION_FT} --weightsInitialiser ${WEIGHTS_INITIALISER} --seed ${SEED}

predict:
	@python3 predict.py --paramsFile ${PARAMS_OUTPUT} --predictFile ${PREDICT_DATASET} --outputFile ${PREDICT_OUTPUT}

clean:
	rm datasets/dataset_train.csv datasets/dataset_predict.csv params.json predictions_output.txt

all: sep train predict


#nice configurations

#LAYERS = 24 24, SEED = 42, BATCHSIZE = 1
# af = sigmoid, wi = random, epoch = 300, lr = 0.0008 Expected acc = 0.9415
# af = sigmoid, wi = heUniform, epoch = 300, lr = 0.0008

#LAYERS = 5 5, SEED = 42, BATCHSIZE = 1
# af = relu, wi = random, epoch = 70, lr = 0.0001

#LAYERS = 24 24, SEED = 42, BATCHSIZE = 30
# af = sigmoid, wi = random, epoch = 300, lr = 0.01

#notes
# relu is more powerful, and can easily cause overfitting, normally used for larger datasets and deeper networks, causes gradients to update more drastically