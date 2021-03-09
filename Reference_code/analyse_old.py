####################
### HOUSEKEEPING ###
####################
from housekeeping_functions import *
import numpy as np

## WHICH SIMULATION?
simulation_name = 'PolyNetwork_large_sparse_2'
print('Simulation name: ' + simulation_name)

## SAVE COPY OF CPP FILE?
copy_cpp_file = 0 #automatically updates to 1 if the simulation you are analysing was running when the first check was made
# only change this to 1 if you are sure the current 'PolyNetwork.cpp' corresponds to the simulation you are analysing

# CHECK UNTIL THE SIMULATION IS FINISHED
# Keeps checking every t seconds until max_t_hours, only executes rest of this code if a simulation is not running
t = 5 #check every t seconds
max_t_hours = 5 #stop checking after max_t_hours
max_t = 60*60*max_t_hours
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

# IMPORT ANALYSIS FUNCTIONS
#it is important that this is done after moving the params, as analysis functions module also imports params
from analysis_functions import *

#PRINT SANITY CHECK
print('Analysing simulation ' + simulation_name + ', initiated at ' + sim_start_time + '...\n\n')

####################
### RASTER PLOTS ###
####################

##### SET HERE ######
raster_plot_bool = 1 ## turns on (1) or off (0) all raster plots
#####################

if raster_plot_bool:

	##### SET HERE ######
	## choose which raster plots you want, (ensure all lists are of equal length)
	test_or_train = ["test", "test"]
	input_or_output = ["input", "output"]
	raster_epochs = [0, 0]
	mintime = 0
	maxtime = 0.1
	#####################

	n_raster_plots = len(raster_epochs)
	if len(test_or_train) == n_raster_plots and len(input_or_output) == n_raster_plots:
		print("Generating raster plots...\n")
		for i in range(n_raster_plots):
			print("     ...for " + input_or_output[i] + " during epoch " + str(raster_epochs[i]) + " of the " + test_or_train[i] + "ing phase...\n")
			plot_raster(test_or_train[i], input_or_output[i], raster_epochs[i], mintime, maxtime)
	else:
		print("Instructions for which raster plots to generate entered incorrectly. None generated.\n\n")

####################
### FIRING RATES ###
####################

#plot_firing_rate_dist("Testing")
#plot_firing_rate_dist("Training")


###############
### WEIGHTS ###
###############

analyse_weights_bool = 0
if analyse_weights_bool:

	print("Analysing weights...\n")
	analyse_weights()