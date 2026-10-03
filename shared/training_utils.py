import copy
from shared.plot import plot_loss, plot_acc

#track loss and acc after each epoch to plot later
def create_history():
	return {
		"train_loss": [],
		"val_loss": [],
		"train_acc": [],
		"val_acc": [],
	}

def record_epoch(history, epoch, epochs, train_loss, val_loss, train_acc, val_acc):
	history["train_loss"].append(train_loss)
	history["val_loss"].append(val_loss)
	history["train_acc"].append(train_acc)
	history["val_acc"].append(val_acc)
	print(f"{epoch}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

def plot_history(history):
	plot_loss(history["train_loss"], history["val_loss"])
	plot_acc(history["train_acc"], history["val_acc"])

#accuracies of the final network, which is the best epoch if early stopping restored it, else the last epoch
def print_final_accuracy(history, early_stop_state):
	if early_stop_state["best_network"] is not None:
		epoch = early_stop_state["best_epoch"]
	else:
		epoch = len(history["train_acc"])

	print(f"Final train accuracy: {history['train_acc'][epoch - 1]:.4f}")
	print(f"Final validation accuracy: {history['val_acc'][epoch - 1]:.4f}")

#early stopping
def create_early_stop_state():
	return {
		"best_val_loss": float("inf"), #start infinitely bad so first epoch is always an improvement
		"best_network": None,
		"best_epoch": 0,
		"epochs_without_improvement": 0,
	}

def early_stop_check(state, network, validation_loss, epoch, patience):
	if patience == -1: # No early stopping 
		return False

	#improved, remember this network and reset the counter
	if validation_loss <= state["best_val_loss"]:
		state["best_val_loss"] = validation_loss
		state["best_network"] = copy.deepcopy(network) #deepcopy works for both numpy and pytorch networks, so later training doesnt change it
		state["best_epoch"] = epoch
		state["epochs_without_improvement"] = 0
		return False

	#no improvement
	state["epochs_without_improvement"] += 1
	return state["epochs_without_improvement"] >= patience
