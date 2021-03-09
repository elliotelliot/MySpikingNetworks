import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.switch_backend('agg') #needed to work on linux
from simulation_params import * #imports variables used in the simulation
#for reference, variables are as follows: simulation_name, training_epochs, display_epochs, timestep, n_ex_layers, n_inh_layers, n_neurons_per_layer, simtime, group_names

#####################
### SANITY CHECKS ###
#####################

##FUNCTION TO GENERATE AXONAL DELAY DISTRIBUTION PLOTS

##FUNCTION TO GENERATE 



####################
### RASTER PLOTS ###
####################

##FUNCTION TO GENERATE A RASTER PLOT FOR A CERTAIN EPOCH FOR INPUT OR OUTPUT
def plot_raster(test_or_train, input_or_output, epoch, mintime, maxtime):

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
	elif input_or_output == "output" or input_or_output == "Output":
		s2 = "Output"
	else:
		print("variable 'input_or_output' should be a string reading either 'input' or 'output'")
		return

	#EXTRACT DATA
	ids = np.array(np.fromfile(sim_output_folder + s1 + "_data/Epoch_" + str(epoch) + "_" + s2 + "_SpikeIDs.bin", dtype=np.int32))
	times = np.array(np.fromfile(sim_output_folder + s1 + "_data/Epoch_" + str(epoch) + "_" + s2 + "_SpikeTimes.bin", dtype=np.float32))


	# SELECT NEURON IDs THAT SPIKED WITHIN TIME FRAME ONLY
	high_pass_mask = times >= mintime
	low_pass_mask = times <= maxtime
	mask = np.where(high_pass_mask*low_pass_mask)
	ids = ids[mask]
	times = times[mask]

	#PLOT RASTER
	plt.figure(figsize = [30, 30])
	plt.scatter(times, ids, s=5)

	#PLOT LINES FOR LAYER BREAKS
	background_input = 0 ########### DELETE THIS LATER
	total_output_neurons = n_neurons_per_layer * (n_ex_layers + n_inh_layers)
	output_layer_ID_breaks = np.arange(0, total_output_neurons-1, n_neurons_per_layer).tolist()
	input_layer_ID_breaks = [0, n_neurons_per_layer]
	x = [mintime, maxtime]
	if input_or_output == "output" or input_or_output == "Output":
		for i in range(len(output_layer_ID_breaks)):
			y = [output_layer_ID_breaks[i]-0.5, output_layer_ID_breaks[i]-0.5]
			if i < n_ex_layers:
				line_label = "^^ Ex layer" + str(i+1)
			else:
				line_label = "^^ Inh layer" + str(i+1)
			plt.plot(x, y, 'b')
			plt.text(maxtime, y[1], line_label)
	else:
		if background_input:
			for i in range(2):
				y = [input_layer_ID_breaks[i], input_layer_ID_breaks[i]]
				plt.plot(x, y, 'b')
			plt.text(maxtime, input_layer_ID_breaks[0]-0.5, "Main input to layer 1 ^")
			plt.text(maxtime, input_layer_ID_breaks[1]-0.5, "Background input to all layers ^")

	#LABEL PLOT
	plt.xlabel("Time (ms)")
	plt.ylabel("Neuron index")
	plt.title(s2 + " neurons raster during " + str(epoch) + " of " + s1 + " phase")
	
	#SAVE AND CLOSE
	plt.savefig(figure_folder + s1 + "_Epoch_" + str(epoch) + "_" + s2 + "_Spikes_Raster")
	plt.close()


####################
### FIRING RATES ###
####################


##FUNCTION TO GENERATE FIRING RATE DISTRIBUTION
def plot_firing_rate_dist(sim_phase_str, mintime = 0, maxtime = simtime):

	#DELETE THIS LATER
	inhibition_bool = 1
	print('doobedoo')
	print('\nAnalysing firing rates during ' + sim_phase_str + ' phase')

	#OPEN FIGURE
	fig, axs = plt.subplots(2, n_ex_layers, figsize=(5*n_ex_layers, 10))

	#CHOOSE CORRECT NUMBER OF EPOCHS
	if sim_phase_str == "Testing":
		n_epochs = display_epochs
	elif sim_phase_str == "Training":
		n_epochs = training_epochs
	else:
		print("Invalid value for sim_phase entered.\n")
		return 0


	for ex_inh in range(1+inhibition_bool):

		for layer in range(n_ex_layers):

			# FIGURE OUT WHICH NEURONS TO ANALYSE
			if ex_inh == 0:
				neuron_type_str = "excitatory"
				start_point = n_neurons_per_layer*layer
				end_point = start_point+n_neurons_per_layer
				neurons = range(start_point, end_point)
				n_neurons = (len(neurons))
			else:
				neuron_type_str = "inhibitory"
				start_point = n_neurons_per_layer*(n_ex_layers + layer)
				end_point = start_point+n_neurons_per_layer
				neurons = range(start_point, end_point)
				n_neurons = (len(neurons))

			print("Generating firing rate distribution for " + neuron_type_str + " layer " + str(layer))
			print('Neuron indices: ' + str(start_point) + ' to ' + str(end_point))
			
			# EXTRACT DATA
			ids = np.array([])
			times = np.array([])
			for epoch in range(n_epochs):
				path = sim_output_folder + sim_phase_str + "_data/Epoch_" + str(epoch) + "_Output"
				ids = np.concatenate([ids, np.fromfile(path + "_SpikeIDs.bin", dtype=np.int32)])
				if mintime != 0 and maxtime != simtime:
					times = np.concatenate([times, np.fromfile(path + "_SpikeTimes.bin", dtype=np.int32)])

			# SELECT NEURON IDs THAT SPIKED WITHIN TIME FRAME ONLY
			if mintime != 0 and maxtime != simtime:
				times = np.array(times)
				high_pass_mask = times > mintime
				low_pass_mask = times < maxtime
				mask = np.multiply(high_pass_mask, low_pass_mask)
				index_mask = np.where(mask)
				ids = ids(index_mask)

			# COMPUTE DISTRIBUTION
			id_cnts = pd.Series(ids).value_counts().to_dict()

			dist = []
			total_spikes = 0
			for neuron in neurons:
				n_spikes_for_this_neuron = id_cnts[neuron]
				total_spikes = total_spikes + n_spikes_for_this_neuron
				rate = n_spikes_for_this_neuron/(simtime*n_epochs)
				dist.append(rate)

			# COMPUTE STATS
			mean = round(total_spikes/(n_neurons*simtime*n_epochs))
			std = round((sum([((x - mean) ** 2) for x in dist]) / len(dist))**0.5)
			print('Mean: ' + str(mean) + '; std: ' + str(std))

			# WRITE TO FILE
			# file_path = figure_folder + "firing_rates.txt"

			# if neuron_type == "input":
			# 	layers_str = "n/a\t\t"
			# else:
			# 	layers_str = str(layers) 

			# if os.path.isfile(file_path):
			# 	firing_rates_file = open(file_path, "a")
			# else:
			# 	firing_rates_file = open(file_path, "w")
			# 	firing_rates_file.write("Neuron Type:\tLayers:\tPhase:\tMin time:\tMax time:\tMean rate:\tStd rate:\n\n")
			# firing_rates_file.write(neuron_type_str + "\t" + layers_str + "\t" + sim_phase + "\t" + str(mintime) + "\t" + str(maxtime) + "\t" + str(mean) + "\t" + str(std) + "\n")
			# firing_rates_file.close

		
			#MAKE FIGURE TITLE
			figure_title = neuron_type_str + " neurons: layer " + str(layer)
		
			#GENERATE PLOT

			[n, bins, patches] = axs[ex_inh, layer].hist(dist, bins=50)
			#plt.xlabel("Firing rate (Hz)")
			#plt.ylabel("Frequency density")
			#plt.text(min(bins), max(n) - max(n)/10, 'Total number of spikes = ' + str(total_spikes))
			#plt.text(min(bins), max(n), 'Average firing rate = ' + str(mean))
			axs[ex_inh, layer].set_title(figure_title)
			
	plt.savefig(figure_folder + sim_phase_str + "_firing_rates")
	plt.close()

	return 0
	

###############
### WEIGHTS ###
###############

def analyse_weights(group_names):

	n_save_weight_groups = len(group_names) #number of groups of synapses (e.g. ff, fb, lat etc)
	x_weights = range(-1, training_epochs) #x axis values for weight evolution plot: -1 (initial weights), then epochs 0:training_epochs-1
	x_learning = range(training_epochs) #x axis values for weight changes: epochs 0:training_epochs-1

	for group in range(n_save_weight_groups): #for each synapse group
		mean_weight = np.zeros([training_epochs+1, 1]) #preallocate mean weight array
		std_weight = np.zeros([training_epochs+1, 1]) #preallocate std weight array
		mean_abs_weight_change = np.zeros([training_epochs, 1]) #preallocate mean weight change array
		std_abs_weight_change = np.zeros([training_epochs, 1]) #preallocate std weight change array
		weights = np.fromfile(sim_output_folder + "Weight_evolution/Initial_" + group_names[group] + "_SynapticWeights.bin", dtype=np.float32) #get initial weights
		mean_weight[0] = np.mean(weights) #assign mean initial weight to first value in mean weight array
		std_weight[0] = np.std(weights) #assign std initial weight to first value in std weight array
		for epoch in range(training_epochs): #for each epoch
			next_weights = np.fromfile(sim_output_folder + "Weight_evolution/Epoch_" + str(epoch) + "_" + group_names[group] + "_SynapticWeights.bin", dtype=np.float32)
			mean_abs_weight_change[epoch] = np.mean(np.abs(np.subtract(next_weights, weights)))
			std_abs_weight_change[epoch] = np.std(np.abs(np.subtract(next_weights, weights)))
			weights = next_weights
			mean_weight[epoch+1] = np.mean(weights)
			std_weight[epoch+1] = np.std(weights)
		plt.figure("weights")
		plt.errorbar(x_weights, mean_weight, yerr = std_weight, xerr = None, label = group_names[group], fmt = 'o')
		plt.figure("weight changes")
		plt.errorbar(x_learning, mean_abs_weight_change, yerr = std_abs_weight_change, xerr = None, label = group_names[group], fmt = 'o')

	plt.figure("weights")
	plt.title("Weight evolution")
	plt.xlabel("Training epoch")
	plt.ylabel("Mean weight")
	plt.legend()
	plt.savefig(figure_folder + "Weight_evo")
	plt.close()

	plt.figure("weight changes")
	plt.title("Learning")
	plt.xlabel("Training epoch")
	plt.ylabel("Mean absolute weight change")
	plt.legend()
	plt.savefig(figure_folder + "Learning")
	plt.close()