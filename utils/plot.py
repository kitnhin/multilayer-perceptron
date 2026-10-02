import matplotlib.pyplot as plot

def plot_loss(train_loss_arr, val_loss_arr):
	epochs = range(1, len(train_loss_arr) + 1)
	plot.plot(epochs, train_loss_arr, label='Training Loss', color='blue')
	plot.plot(epochs, val_loss_arr, label='Validation Loss', color='orange')
	plot.xlabel('Epoch')
	plot.ylabel('Loss')
	plot.title('Training and Validation Loss')
	plot.legend()
	plot.show()

def plot_acc(train_acc_arr, val_acc_arr):
	epochs = range(1, len(train_acc_arr) + 1)
	plot.plot(epochs, train_acc_arr, label='Training Accuracy', color='blue')
	plot.plot(epochs, val_acc_arr, label='Validation Accuracy', color='orange')
	plot.xlabel('Epoch')
	plot.ylabel('Accuracy')
	plot.title('Training and Validation Accuracy')
	plot.legend()
	plot.show()