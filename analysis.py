import numpy as np
import pandas as pd #pandas has multiple functions, including providing 'data_frame' objects that can be used for visualizing and analyzing data
from matplotlib import pyplot as plt
plt.switch_backend('agg') #needed to work on linux

from simulation_params import * #imports variables used in the simulation
#for reference, variable are as follows: simulation_name, training_epochs, display_epochs, timestep, n_ex_layers, n_inh_layers, n_neurons_per_layer, simtime

def plot_raster(test_or_train, input_or_output, epoch):
	
	print(input_or_output)

	#SET STRINGS FOR DATA READING/LABELLING
	if test_or_train == "test" or test_or_train == "Test":
		s1 = "Testing"
	elif test_or_train == "train" or test_or_train == "Train":
		s1 = "Training"
	else:
		print("variable 'test_or_train' should be a string reading either 'test' or 'train'")
		return

	if input_or_output == "input" or input_or_output == "Input":
		s2 = "Input"
		print(s2)
	elif input_or_output == "output" or input_or_output == "Output":
		s2 = "Output"
		print(s2)
	else:
		print("variable 'input_or_output' should be a string reading either 'input' or 'output'")
		return

	#EXTRACT DATA
	df = pd.DataFrame( #NB Pandas will generate a 'data-frame', where each row in this case has a name (ids or times), and the columns in those rows contain the values of interest
  	data = {
      "ids": np.fromfile("../Outputs/" + simulation_name + "/" + s1 + "_data/Epoch_" + str(epoch) + "_" + s2 + "_SpikeIDs.bin", dtype=np.int32),
      "times": np.fromfile("../Outputs/" + simulation_name + "/" + s1 + "_data/Epoch_"+ str(epoch) + "_" + s2 + "_SpikeTimes.bin", dtype=np.float32),
  	}
	)

	#PLOT RASTER
	plt.figure(figsize = [12, 9])
	plt.scatter(df["times"], df["ids"], s=5)

	#PLOT LINES FOR LAYER BREAKS
	if input_or_output == "output" or input_or_output == "Output":
		total_neurons = n_neurons_per_layer*(n_ex_layers+n_inh_layers)
		layer_ID_breaks = np.arange(0, total_neurons-1, n_neurons_per_layer).tolist()
		layer_ID_breaks = [i - 0.5 for i in layer_ID_breaks]
		x = [0, simtime]
		for j in range(len(layer_ID_breaks)):
			y = [layer_ID_breaks[j], layer_ID_breaks[j]]
			if j < n_ex_layers:
				line_label = "^^ Ex layer" + str(j+1)
			else:
				line_label = "^^ Inh layer" + str(j+1)
			plt.plot(x, y, 'b')
			plt.text(simtime, y[1], line_label)

	#LABEL AXES
	plt.xlabel("Time (ms)")
	plt.ylabel("Neuron index")
	
	#SAVE AND CLOSE
	plt.savefig("Figures/" + s1 + "_Epoch_" + str(epoch) + "_" + s2 + "_Spikes_Raster")
	plt.close()
	

test_or_train = "test"
input_or_output = "input"
print(input_or_output)
epoch = 0
plot_raster(test_or_train, input_or_output, epoch)


#mask = stim1_df["times"] <1.0 #Restrict plotted spikes to a particular time period
	#mask = ((stim1_df["ids"] > 0) & (stim1_df["ids"] <= 1024)) #Restrict plotted spikes to a particular layer

