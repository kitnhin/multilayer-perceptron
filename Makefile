#file paths
TRAIN_DATASET = datasets/dataset_train.csv
TEST_DATASET = datasets/dataset_test.csv
VALIDATION_DATASET = datasets/dataset_validation.csv
GIVEN_DATASET = datasets/data.csv

#source folders
NUMPY_DIR = numpy-ver
PYTORCH_DIR = pytorch-ver
SHARED_DIR = shared

#output files
NP_PARAMS_OUTPUT = ${NUMPY_DIR}/params.json
NP_PREDICT_OUTPUT = ${NUMPY_DIR}/predictions_output.txt
PT_PARAMS_OUTPUT = ${PYTORCH_DIR}/model.pt
PT_PREDICT_OUTPUT = ${PYTORCH_DIR}/predictions_output.txt

#separation settings
TRAIN_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.15 # test dataset gets the rest (1 - TRAIN_PERCENTAGE - VALIDATION_PERCENTAGE)
SEED = 42 #SEED = -1 means no seed, random

#training configs
EPOCHS = 300
LAYERS = 24 24
LEARNING_RATE = 0.001
BATCH_SIZE = 1 #put 1 for SGD
ACTIVATION_FT = sigmoid #sigmoid or relu
WEIGHTS_INITIALISER = random #heUniform or random
PATIENCE = 10 #early stopping: stop if val loss doesnt improve for this many epochs, -1 to disable


sep:
	@PYTHONPATH=. uv run python ${SHARED_DIR}/separate.py --givenFile ${GIVEN_DATASET} --trainFile ${TRAIN_DATASET} --validationFile ${VALIDATION_DATASET} --testFile ${TEST_DATASET} \
	--trainPercentage ${TRAIN_PERCENTAGE} --validationPercentage ${VALIDATION_PERCENTAGE} --seed ${SEED}

np_train:
	@PYTHONPATH=. uv run python ${NUMPY_DIR}/train.py --trainFile ${TRAIN_DATASET} --outputFile ${NP_PARAMS_OUTPUT} --layer ${LAYERS} --epochs ${EPOCHS} --learningRate ${LEARNING_RATE} --validationFile ${VALIDATION_DATASET} \
	--batchSize ${BATCH_SIZE} --activationFt ${ACTIVATION_FT} --weightsInitialiser ${WEIGHTS_INITIALISER} --seed ${SEED} --patience ${PATIENCE}

np_predict:
	@PYTHONPATH=. uv run python ${NUMPY_DIR}/predict.py --paramsFile ${NP_PARAMS_OUTPUT} --testFile ${TEST_DATASET} --outputFile ${NP_PREDICT_OUTPUT}

pt_train:
	@PYTHONPATH=. uv run python ${PYTORCH_DIR}/train.py --trainFile ${TRAIN_DATASET} --outputFile ${PT_PARAMS_OUTPUT} --layer ${LAYERS} --epochs ${EPOCHS} --learningRate ${LEARNING_RATE} --validationFile ${VALIDATION_DATASET} \
	--batchSize ${BATCH_SIZE} --activationFt ${ACTIVATION_FT} --weightsInitialiser ${WEIGHTS_INITIALISER} --seed ${SEED} --patience ${PATIENCE}

pt_predict:
	@PYTHONPATH=. uv run python ${PYTORCH_DIR}/predict.py --paramsFile ${PT_PARAMS_OUTPUT} --testFile ${TEST_DATASET} --outputFile ${PT_PREDICT_OUTPUT}

clean:
	rm -f ${TRAIN_DATASET} ${VALIDATION_DATASET} ${TEST_DATASET} ${NP_PARAMS_OUTPUT} ${NP_PREDICT_OUTPUT} ${PT_PARAMS_OUTPUT} ${PT_PREDICT_OUTPUT}

all: sep np_train np_predict


#nice configurations

#LAYERS = 24 24, SEED = 42, BATCHSIZE = 1, af = sigmoid, wi = random, epoch = 300, lr = 0.001
# Expected acc = 0.9419 (Numpy), 0.9535 (Pytorch)


#LAYERS = 5 5, SEED = 42, BATCHSIZE = 1
# af = relu, wi = random, epoch = 70, lr = 0.0001

#LAYERS = 24 24, SEED = 42, BATCHSIZE = 30
# af = sigmoid, wi = random, epoch = 300, lr = 0.01

#notes
# relu is more powerful, and can easily cause overfitting, normally used for larger datasets and deeper networks, causes gradients to update more drastically
# PYTHONPATH=. is to specify the project root dir so modules can be imported correctly