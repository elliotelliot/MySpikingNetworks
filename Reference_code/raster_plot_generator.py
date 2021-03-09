#RASTER SETTINGS
raster_epochs = 0
raster_time_window = [0, 0.1]

####################
### RASTER PLOTS ###
####################

str_list = ['Input', 'Output']
mintime = raster_time_window[0]
maxtime = raster_time_window[1]

for epoch in raster_epochs:

	for input_or_output in range(1):


		#EXTRACT DATA
		ids = np.array(np.fromfile(sim_output_folder + sim_phase_str + "_data/Epoch_" + str(epoch) + "_" + str_list(input_or_output) + "_SpikeIDs.bin", dtype=np.int32))
		times = np.array(np.fromfile(sim_output_folder + sim_phase_str + "_data/Epoch_" + str(epoch) + "_" + str_list(input_or_output) + "_SpikeTimes.bin", dtype=np.float32))

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
		x = [mintime, maxtime]
		if input_or_output:
			total_output_neurons = n_neurons_per_layer * (n_ex_layers + n_inh_layers)
			output_layer_ID_breaks = np.arange(0, total_output_neurons-1, n_neurons_per_layer).tolist()
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
				input_layer_ID_breaks = [0, n_neurons_per_layer]
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