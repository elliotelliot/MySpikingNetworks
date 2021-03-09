import os.path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.switch_backend('agg') #needed to work on linux

## sudo scontrol update NodeName=oftserve State=Resume

####################
### HOUSEKEEPING ###
####################
from housekeeping_functions import *
import numpy as np

## WHICH SIMULATION?
simulation_name = 'PolyNetwork_topology'

## SAVE COPY OF CPP FILE?
copy_cpp_file = 0 #automatically updates to 1 if the simulation you are analysing was running when the first check was made
# only change this to 1 if you are sure the current 'PolyNetwork.cpp' corresponds to the simulation you are analysing

# CHECK UNTIL THE SIMULATION IS FINISHED
# Keeps checking every t seconds until max_t_hours, only executes rest of this code if a simulation is not running
t = 5 #check every t seconds
max_t = 120 #stop checking after max_t
copy_cpp_file = check_complete(simulation_name, t, max_t)

if copy_cpp_file:

	# SAVE A COPY OF SIMULATION CPP CODE
	# saves into output directory 
	save_sim_cpp_copy(simulation_name)

# MOVE PARAMETERS
# moves parameters from sim output directory to current directory, overwriting previously used parameter file
# this is so they can be imported
move_params(simulation_name)

# IMPORT VARIABLES
from simulation_params import * #imports variables used in the simulation
#for reference, variables are as follows: simulation_name, sim_output_folder, figure_folder, training_epochs, display_epochs, timestep, n_ex_layers, n_inh_layers, n_neurons_per_layer, simtime, group_names

#PRINT SANITY CHECK
print('Analysing simulation ' + simulation_name + ', initiated at ' + sim_start_time + '...\n\n')



######################
### QUICK SETTINGS ###
######################

# DATA HANDLING
keep_data = 0 # Once data is extracted, do you want to save it?
save_in_new_folder = 0 # Do you want to save figures and extracted data in a new folder?
new_folder_name = "foo/"


# Make raster plot?
raster_plots = 1

# Make firing rate plots?
firing_rate_plots = 1

# Make firing rate through time plots?
firing_rate_through_time = 0

# Make weight evolution plots?
weight_evolution = 0



#####################
### SANITY CHECKS ###
#####################

##FUNCTION TO GENERATE AXONAL DELAY DISTRIBUTION PLOTS

##FUNCTION TO GENERATE 

#################################
### CALCULATE SOME PARAMETERS ###
#################################


n_inh_layers = n_ex_layers*inhibition_bool
n_layers = n_ex_layers+n_inh_layers

n_inh_neurons = n_inh_layers*n_inh_neurons_per_layer
n_ex_neurons = n_ex_layers*n_ex_neurons_per_layer
n_neurons = n_inh_neurons+n_ex_neurons

# note: n_input_neurons is saved from cpp file
# note: 1 input layer

if save_in_new_folder:
	figure_folder = figure_folder + new_folder_name

if not os.path.exists(figure_folder):
    os.mkdir(figure_folder)

####################
### EXTRACT DATA ###
####################

#EXTRACTS AND STORES MAIN NEURON SPIKE TIMES INDEXED BY EPOCH BY NEURON
# Stores in an epoch-by-epoch list of neuron-by-neuron lists of lists of spike times
# (reminder: layer ids go excitatory layers 0 through n-1, then inhibitory layers n through 2n-1, with n being number of layers)

if save_in_new_folder:
	data_file_path = figure_folder + "extracted_main_data.txt"
else:
	data_file_path = sim_output_folder + "extracted_main_data.txt"


if os.path.isfile(data_file_path) == False:

	print("\nExtracting data...")

	spike_times_by_epoch_by_neuron = [] # Declare list for storing spike times indexed by epoch by neuron

	for epoch in range(n_epochs): # Loop through all epochs

		path = sim_output_folder + "Main_neurons/Epoch_" + str(epoch) # File path for main neurons for this epoch

		print(path)

		ids = np.fromfile(path + "_SpikeIDs.bin", dtype=np.int32) # Extract neuron IDs
		times = np.fromfile(path + "_SpikeTimes.bin", dtype=np.float32) # Extract spike times

		times_for_this_epoch = [] # Declare temp neuron-by-neuron list of lists of spike times for this epoch

		for neuron_id in range(n_neurons): # Loop through each neuron

			if np.any(ids == neuron_id):
				times_for_this_neuron = times[np.where(ids == neuron_id)] # Create array of times when this neuron spiked in this epoch
				times_for_this_neuron = times_for_this_neuron.tolist()
			else:
				times_for_this_neuron = []

			times_for_this_epoch.append(times_for_this_neuron) # Append this array to neuron-by-neuron list for this epoch

		
		spike_times_by_epoch_by_neuron.append(times_for_this_epoch) # Add neuron-by-neuron list of lists of times to master list of lists of lists

	if keep_data:

		print("\nWriting data to file...")

		with open(data_file_path, 'w') as output:
			output.write(json.dumps(spike_times_by_epoch_by_neuron))

else:

	print("\nLoading data...")
	with open(data_file_path, 'r') as output:
		spike_times_by_epoch_by_neuron = json.loads(output.read())


# TO ADD: save this list in sim output folder?? Or generate this list while running simulation??

############################
### RASTER PLOT FUNCTION ###
############################

##FUNCTION TO GENERATE A RASTER PLOT FOR A CERTAIN EPOCH FOR INPUT OR OUTPUT
def plot_raster(epoch, mintime = 0, maxtime = simtime, input_neurons = 0):

	if input_neurons == 0:
		path = sim_output_folder + "/Main_neurons/"
		title_string = "Main"
	else:
		path = sim_output_folder + "/Input_neurons/"
		title_string = "Input"

	#EXTRACT DATA
	ids = np.array(np.fromfile(path + "Epoch_" + str(epoch) + "_SpikeIDs.bin", dtype=np.int32))
	times = np.array(np.fromfile(path + "Epoch_" + str(epoch) + "_SpikeTimes.bin", dtype=np.float32))


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
	if input_neurons == 0:
		breaks1 = np.arange(0, n_ex_neurons-1, n_ex_neurons_per_layer).tolist()
		breaks2 = np.arange(n_ex_neurons, n_neurons-1, n_inh_neurons_per_layer).tolist()
		breaks = breaks1 + breaks2
		x = [mintime, maxtime]
		for i in range(len(breaks)):
			y = [breaks[i]-0.5, breaks[i]-0.5]
			if i < n_ex_layers:
				line_label = "^^ Ex layer" + str(i+1)
			else:
				line_label = "^^ Inh layer" + str(i+1)
			plt.plot(x, y, 'b')
			plt.text(maxtime, y[1], line_label)

	#LABEL PLOT
	plt.xlabel("Time (ms)")
	plt.ylabel("Neuron index")
	plt.title(title_string + " neurons raster during epoch " + str(epoch))
	
	#SAVE AND CLOSE
	plt.savefig(figure_folder + "Epoch_" + str(epoch) + "_Spikes_Raster_" + title_string)
	plt.close()



##################################
### FIRING RATE PLOT FUNCTIONS ###
##################################

def generate_main_firing_rate_plot(epoch_range, figure_name):


	# GENERATE LAYER-BY-LAYER LIST OF ARRAYS OF RATES FOR THIS EPOCH RANGE
	rates_by_layer = [np.array([])] # Declare list for storing rates indexed by layer and initialise first element to empty numpy array
	for i in range(1, n_layers): # For each layer...
		rates_by_layer.append(np.array([])) # ...append another empty numpy array to list for storing spike times indexed by layer
	for neuron_id in range(n_neurons): # Cycle through all neurons
		if neuron_id < n_ex_neurons: # If this is an excitatory neuron...
			layer_id = neuron_id/n_ex_neurons_per_layer # ...calculate which layer this neuron is in
		else: # If this is an inhibitory neuron...
			layer_id = ((neuron_id-n_ex_neurons)/n_inh_neurons_per_layer) + n_ex_layers # ...calculate which layer this neuron is in
		for epoch in range(epoch_range[0], epoch_range[1]): # Cycle through epochs in specified range			
			spike_times = spike_times_by_epoch_by_neuron[epoch][neuron_id] # Access spike times for this epoch for this neuron
			rate = len(spike_times)/simtime # Calculate rate for this epoch for this neuron
			rates_by_layer[layer_id] = np.append(rates_by_layer[layer_id], rate) # Add rate this neuron for this epoch to array for this layer

	
	# SET FIGURE SETTINGS AND OPEN PLOT
	bins_edge_list = range(0, int(1/0.002+1), 5) 
	inches_per_plot = 5
	fig, axs = plt.subplots(1+inhibition_bool, n_ex_layers, figsize=(inches_per_plot*n_ex_layers, inches_per_plot*(inhibition_bool+1)))
	neuron_type_str_list = ["excitatory", "inhibitory"]
	
	n_epochs = epoch_range[1] - epoch_range[0] + 1
	
	# GET RATES FOR EACH LAYER
	for layer_id in range(n_layers): # Cycle through layers
		
		rates = rates_by_layer[layer_id] # Get array of rates for this layer

		# SET SUBPLOT COORDINATES
		ex_inh = layer_id/n_ex_layers # =0 if ex layer, =1 if inh layer
		x_coord = layer_id - n_layers*(ex_inh)
		if inhibition_bool: # If there are inhibitory layers, we need a 2D array of subplots...
			sbplt = axs[ex_inh, x_coord] #...set 2D coordinates
		else: # If there are no inhibitory layers, we need a 1D array of subplots...
			sbplt = axs[x_coord] #...set 1D coordinate

		# PLOT SUBPLOT
		[n, bins, patches] = sbplt.hist(rates, bins=bins_edge_list)

		# SET AXES LABELS
		if ex_inh == 0: # If we are on the bottom row of subplots...
			sbplt.set(xlabel = "Firing rate (Hz)") # ...set x-axis label
		if x_coord == 0: #If we are on the leftmost column of subplots...
			sbplt.set(ylabel = "Frequency density") # ...set y-axis label	

		# SET SUBPLOT TITLE
		subplot_title = neuron_type_str_list[ex_inh] + " neurons: layer " + str(layer_id) # Make subplot title
		sbplt.set_title(subplot_title) # Set subplot title


		# COMPUTE STATS
		n_neurons_per_layer = ex_inh*n_inh_neurons_per_layer - (ex_inh-1)*n_ex_neurons_per_layer # Calculate number of neurons per layer (-(ex_inh)-1)=1 if ex layer, 0 if inh layer)
		mean = round(sum(rates)/(n_neurons_per_layer*n_epochs)) # Calculate mean rate for this layer
		std = round((sum([((x - mean) ** 2) for x in rates]) / len(rates))**0.5) # Calculate std rate for this layer
		print('Layer ' + str(layer_id) + ' -- Mean: ' + str(mean) + '; std: ' + str(std)) # Print stats
	

	#SAVE AND CLOSE
	plt.savefig(figure_folder + figure_name)
	plt.close()


def generate_input_firing_rate_plot(epoch_range, figure_name):

	# EXTRACT DATA
	ids = np.array([])
	times = np.array([])
	for epoch in range(epoch_range[0], epoch_range[1]+1):
		path = sim_output_folder + "Input_neurons/Epoch_" + str(epoch)
		ids = np.concatenate([ids, np.fromfile(path + "_SpikeIDs.bin", dtype=np.int32)])
		times = np.concatenate([times, np.fromfile(path + "_SpikeTimes.bin", dtype=np.int32)])

	# COMPUTE DISTRIBUTION
	n_epochs = epoch_range[1] - epoch_range[0] + 1
	id_cnts = pd.Series(ids).value_counts().to_dict()
	input_dist = []
	total_spikes = np.zeros(2)
	for neuron in range(0, n_input_neurons):
		n_spikes_for_this_neuron = id_cnts[neuron]
		rate = n_spikes_for_this_neuron/(simtime*n_epochs)
		input_dist.append(rate)
		total_spikes[0] = total_spikes[0] + n_spikes_for_this_neuron

	#COMPUTE STATS
	mean = round(total_spikes[0]/(n_input_neurons*simtime*n_epochs))
	std = round((sum([((x - mean) ** 2) for x in input_dist]) / len(input_dist))**0.5)
	print("Main input mean rate = " + str(mean) + ", std = " + str(std))
	
	#GENERATE FIGURE
	fig = plt.hist(input_dist, bins = 50)

	#SAVE AND CLOSE
	plt.savefig(figure_folder + figure_name)
	plt.close()


####################
### RUN ANALYSES ###
####################

if raster_plots:
	
	print("\nGenerating raster plots...")
	plot_raster(0)



if firing_rate_plots:

	print("\nGenerating input firing rate plot...")
	generate_input_firing_rate_plot([0, (n_epochs-1)], "Input_firing_rate_plot")

	# print("\nGenerating pre-training firing rate plots...")
	# generate_main_firing_rate_plot([0, 9], "Pre-train_firing_rates")
	# print("\nGenerating training firing rate plots...")
	# generate_main_firing_rate_plot([10, 19], "Training_firing_rates")
	# print("\nGenerating post-training firing rate plots...")
	# generate_main_firing_rate_plot([20, 29], "Post-training_firing_rates")

	print("\nGenerating main firing rate plots...")
	generate_main_firing_rate_plot([0, (n_epochs-1)], "Main_firing_rates")



##############################
### FIRING RATES OVER TIME ###
##############################


# print("\nGenerating firing rates over time plots...")

# n_time_bins = 10
# time_bin_width = simtime/n_time_bins

# print("Bin width = " + str(time_bin_width))

# mean_time_series = []

# for neuron in range(total_main_neurons):

# 	sum_time_series = np.zeros(n_time_bins) # Initialise sum_time_series for this neuron - sum accross epochs

# 	for epoch in range(n_epochs):

# 		spike_times = np.array(spike_times_by_epoch_by_neuron[epoch][neuron]) # Spike times for this epoch and neuron (as numpy array)

# 		time_series = np.zeros(n_time_bins) # Initialise time_series for this epoch and neuron

# 		if spike_times.any(): # Populate time_series if there are any spike times for this neuron for this epoch - values left as zeros if not
			
# 			for i in range(n_time_bins): # For each time bin

# 				spikes = [spike_times[(spike_times >= time_bin_width*i) & (spike_times < time_bin_width*(i+1))]] # Find spikes within this time bin

# 				rate = np.size(spikes)/time_bin_width # Calculate rate for this time bin

# 				time_series[i] = rate # Enter into time_series

# 		sum_time_series = sum_time_series + time_series # Add to running tally

# 	mean_time_series[neuron] = sum_time_series/n_epochs


# print(mean_time_series)

### Need to devise some kind of statistical test for determining which time series are uniform, and only look at those that are not


# ###############
# ### WEIGHTS ###
# ###############

# def analyse_weights(group_names):

# 	n_save_weight_groups = len(group_names) #number of groups of synapses (e.g. ff, fb, lat etc)
# 	x_weights = range(-1, training_epochs) #x axis values for weight evolution plot: -1 (initial weights), then epochs 0:training_epochs-1
# 	x_learning = range(training_epochs) #x axis values for weight changes: epochs 0:training_epochs-1

# 	for group in range(n_save_weight_groups): #for each synapse group
# 		mean_weight = np.zeros([training_epochs+1, 1]) #preallocate mean weight array
# 		std_weight = np.zeros([training_epochs+1, 1]) #preallocate std weight array
# 		mean_abs_weight_change = np.zeros([training_epochs, 1]) #preallocate mean weight change array
# 		std_abs_weight_change = np.zeros([training_epochs, 1]) #preallocate std weight change array
# 		weights = np.fromfile(sim_output_folder + "Weight_evolution/Initial_" + group_names[group] + "_SynapticWeights.bin", dtype=np.float32) #get initial weights
# 		mean_weight[0] = np.mean(weights) #assign mean initial weight to first value in mean weight array
# 		std_weight[0] = np.std(weights) #assign std initial weight to first value in std weight array
# 		for epoch in range(training_epochs): #for each epoch
# 			next_weights = np.fromfile(sim_output_folder + "Weight_evolution/Epoch_" + str(epoch) + "_" + group_names[group] + "_SynapticWeights.bin", dtype=np.float32)
# 			mean_abs_weight_change[epoch] = np.mean(np.abs(np.subtract(next_weights, weights)))
# 			std_abs_weight_change[epoch] = np.std(np.abs(np.subtract(next_weights, weights)))
# 			weights = next_weights
# 			mean_weight[epoch+1] = np.mean(weights)
# 			std_weight[epoch+1] = np.std(weights)
# 		plt.figure("weights")
# 		plt.errorbar(x_weights, mean_weight, yerr = std_weight, xerr = None, label = group_names[group], fmt = 'o')
# 		plt.figure("weight changes")
# 		plt.errorbar(x_learning, mean_abs_weight_change, yerr = std_abs_weight_change, xerr = None, label = group_names[group], fmt = 'o')

# 	plt.figure("weights")
# 	plt.title("Weight evolution")
# 	plt.xlabel("Training epoch")
# 	plt.ylabel("Mean weight")
# 	plt.legend()
# 	plt.savefig(figure_folder + "Weight_evo")
# 	plt.close()

# 	plt.figure("weight changes")
# 	plt.title("Learning")
# 	plt.xlabel("Training epoch")
# 	plt.ylabel("Mean absolute weight change")
# 	plt.legend()
# 	plt.savefig(figure_folder + "Learning")
# 	plt.close()