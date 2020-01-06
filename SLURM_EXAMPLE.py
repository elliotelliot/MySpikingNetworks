import os, commands

print "hello world"

def create_shell_script_and_sbatch(shell_file_name, command_for_shell_file, dependency_job_id=-1):
		
	f = open(shell_file_name,"w+")
	f.write("#!/bin/bash\n" + command_for_shell_file)
	f.close()

	dependency_string = ""
	if (dependency_job_id > -1):
		dependency_string = "--depend=afterany:" + str(dependency_job_id)

	command_prefix = "sbatch --gres=gpu:1 "
	print command_prefix + dependency_string + " ./" + shell_file_name
	status, job_id = commands.getstatusoutput(command_prefix + dependency_string + " ./" + shell_file_name)
	print status, job_id


shell_file_name =  "OP_" + str(0)
command_for_shell_file = "./Build/Examples/SimpleExample"

create_shell_script_and_sbatch(shell_file_name, command_for_shell_file)



	

	# def run_training_and_testing(self, model_proto_post_optimisation_save_file_name_and_directory):

	# 	training_dependency_job_id = -1
	# 	testing_dependency_job_id = -1

	# 	testing_and_training_directory = self.experiment_output_directory + "TestingAndTraining/"
	# 	self.directory_helper.create_and_make_sure_path_exists(testing_and_training_directory)

	# 	for experiment_stage_index in range(self.number_of_experiment_stages):

	# 		optimised_model_proto = ji_proto_2_pb2.Flvsm()
	# 		self.proto_helper.read_proto_file_into_proto_object(model_proto_post_optimisation_save_file_name_and_directory, optimised_model_proto)

	# 		number_of_stimulus_subsets = len(self.experiment_proto.experiment_stages[experiment_stage_index].stimulus_subsets)
	# 		stimulus_manager = StimulusManager.StimulusManager()

	# 		if (optimised_model_proto.input_protocol_type == ji_proto_2_pb2.STANDARD_STIMULI_WITH_MEAN_FIRST_SPIKE_TIME_INPUT_PROTOCOL_TYPE):

	# 			array_of_standard_stimuli_subset_directories = []

	# 			for stimulus_subset_index in range(number_of_stimulus_subsets):
	# 				stimulus_subset_proto_object = self.experiment_proto.experiment_stages[experiment_stage_index].stimulus_subsets[stimulus_subset_index]
	# 				standard_stimulus_subset_directory = stimulus_manager.setup_standard_stimuli(optimised_model_proto, self.experiment_output_directory, stimulus_subset_index, stimulus_subset_proto_object.predefined_stimulus_type, stimulus_subset_proto_object.object_stimulus_split_stimuli_gap_blocks, stimulus_subset_proto_object.standard_number_of_neurons_in_block, stimulus_subset_proto_object.include_partly_invisible_stimuli)
	# 				array_of_standard_stimuli_subset_directories.append(standard_stimulus_subset_directory)


	# 		experiment_stage_testing_and_training_directory = testing_and_training_directory + "ExperimentStage" + str(experiment_stage_index) + "/"
	# 		self.directory_helper.create_and_make_sure_path_exists(experiment_stage_testing_and_training_directory)

	# 		self.proto_helper.save_proto_params_to_file(experiment_stage_testing_and_training_directory + "OptimisedModelProto.txt", optimised_model_proto)

	# 		for training_phase_index in range(-1, self.experiment_proto.number_of_training_phases):

	# 			# print 'Training Phase: ' + str(training_phase_index)

	# 			training_phase_directory = experiment_stage_testing_and_training_directory + "TrainingPhase" + str(training_phase_index) + "/"
	# 			self.directory_helper.create_and_make_sure_path_exists(training_phase_directory)


	# 			# Training
	# 			if (training_phase_index > -1):

	# 				finished_training_for_training_phase_file = training_phase_directory + "FINISHED_TRAINING_SIMULATION.txt"
	# 				if ((self.experiment_proto.skip_if_done_training == False) | (os.path.isfile(finished_training_for_training_phase_file) == 0)):
	# 					if (os.path.isfile(finished_training_for_training_phase_file) == 1):
	# 						os.remove(finished_training_for_training_phase_file)

	# 					training_params = ji_proto_2_pb2.NewTestAndTrainingParams()
	# 					training_params.new_test_or_training_type = ji_proto_2_pb2.NEW_TEST_OR_TRAINING_TYPE_TRAINING
	# 					training_params.data_output_directory = training_phase_directory
	# 					training_proto_directory_and_name = training_phase_directory + "TrainingProto.txt"
	# 					training_params.save_weights = True;
	# 					training_params.directory_for_weights_save = training_phase_directory
	# 					training_params.load_weights = True;
	# 					training_params.directory_for_weights_load = experiment_stage_testing_and_training_directory + "TrainingPhase" + str(training_phase_index - 1) + "/"
	# 					training_params.number_of_sub_phases = self.experiment_proto.number_of_TRAINING_sub_phases_per_training_phase

	# 					if (optimised_model_proto.input_protocol_type == ji_proto_2_pb2.STANDARD_STIMULI_WITH_MEAN_FIRST_SPIKE_TIME_INPUT_PROTOCOL_TYPE):
	# 						training_params.standard_stimulus_subset_directory = array_of_standard_stimuli_subset_directories[0]
	# 					elif (optimised_model_proto.input_protocol_type == ji_proto_2_pb2.SPIKES_FROM_FILE_INPUT_PROTOCOL_TYPE):
	# 						training_params.spikes_from_file_particular_test_or_training_directory = optimised_model_proto.spikes_from_file_data_root + "StimulusSubset" + str(0) + "/Training/Phase" + str(training_phase_index) + "/"
	# 						# print training_params.spikes_from_file_particular_test_or_training_directory
	# 						training_params.spikes_from_file_protocol_number_of_stimuli = len([name for name in os.listdir(training_params.spikes_from_file_particular_test_or_training_directory + "SubPhase0/") if os.path.isdir(os.path.join(training_params.spikes_from_file_particular_test_or_training_directory + "SubPhase0/", name))])
	# 						stimulus_manager.stimulus_indices = range(training_params.spikes_from_file_protocol_number_of_stimuli)
							
	# 					stimulus_manager.create_stimulus_indices_by_phase_for_test_training_proto(training_params.lists_of_stimulus_indices_by_phase, self.experiment_proto.number_of_TRAINING_sub_phases_per_training_phase, 0, True)

	# 					self.proto_helper.save_proto_params_to_file(training_proto_directory_and_name, training_params)
						

	# 					shell_file_name = training_phase_directory + "P" + str(self.independent_variable_combination_index) + "_" + str(training_phase_index)
	# 					command_for_shell_file = "./Executables/TestOrTrain " + optimised_model_proto.model_proto_post_optimisation_save_file_name_and_directory + " " + training_proto_directory_and_name + self.command_suffix
	# 					job_id = self.create_shell_script_and_sbatch(shell_file_name, command_for_shell_file, training_dependency_job_id)
	# 					training_dependency_job_id = job_id
	# 					testing_dependency_job_id = training_dependency_job_id



	# 				if (training_phase_index == self.experiment_proto.number_of_training_phases - 1):
	# 					self.experiment_proto.experiment_stages[experiment_stage_index].end_of_stage_weights_file_path = training_phase_directory
	# 					self.proto_helper.save_proto_params_to_file(self.experiment_test_file_and_directory, self.experiment_proto)


	# 			# Testing
	# 			for stimulus_subset_index in range(number_of_stimulus_subsets):

	# 				stimulus_subset_proto_object = self.experiment_proto.experiment_stages[experiment_stage_index].stimulus_subsets[stimulus_subset_index]

	# 				if (stimulus_subset_proto_object.number_of_TESTING_sub_phases_per_training_phase > 0):

	# 					stimulus_subset_directory = training_phase_directory + 'StimulusSubset' + str(stimulus_subset_index) + "/"
	# 					self.directory_helper.create_and_make_sure_path_exists(stimulus_subset_directory)

	# 					finished_testing_for_training_phase_file = stimulus_subset_directory + "FINISHED_TESTING_SIMULATION.txt"


	# 					if ((self.experiment_proto.skip_if_done_testing == False) | (os.path.isfile(finished_testing_for_training_phase_file) == 0)):
	# 						if (os.path.isfile(finished_testing_for_training_phase_file) == 1):
	# 							os.remove(finished_testing_for_training_phase_file)

	# 						testing_params = ji_proto_2_pb2.NewTestAndTrainingParams()
	# 						testing_params.new_test_or_training_type = ji_proto_2_pb2.NEW_TEST_OR_TRAINING_TYPE_TEST
	# 						testing_params.data_output_directory = stimulus_subset_directory

	# 						test_proto_directory_and_name = stimulus_subset_directory + "TestingProto.txt"

	# 						if (training_phase_index == -1):
	# 							testing_params.save_weights = True;
	# 							testing_params.directory_for_weights_save = training_phase_directory
	# 							testing_params.save_connectivity = True;
	# 							testing_params.directory_for_connectivity_save = experiment_stage_testing_and_training_directory
	# 							if (experiment_stage_index > 0):
	# 								testing_params.previous_layer_by_layer_weight_load = True
	# 								testing_params.load_weights = True;
	# 								testing_params.directory_for_weights_load = self.experiment_proto.experiment_stages[experiment_stage_index-1].end_of_stage_weights_file_path # Should put into optimisation code eventually + a proto

	# 						elif (training_phase_index > -1):
	# 							testing_params.load_weights = True;
	# 							testing_params.directory_for_weights_load = training_phase_directory
	# 							# print 'testing_params.directory_for_weights_load: ' + str(testing_params.directory_for_weights_load)


	# 						testing_params.number_of_sub_phases = stimulus_subset_proto_object.number_of_TESTING_sub_phases_per_training_phase

	# 						if (optimised_model_proto.input_protocol_type == ji_proto_2_pb2.STANDARD_STIMULI_WITH_MEAN_FIRST_SPIKE_TIME_INPUT_PROTOCOL_TYPE):
	# 							testing_params.standard_stimulus_subset_directory = array_of_standard_stimuli_subset_directories[stimulus_subset_index]
	# 						elif (optimised_model_proto.input_protocol_type == ji_proto_2_pb2.SPIKES_FROM_FILE_INPUT_PROTOCOL_TYPE):
	# 							testing_params.spikes_from_file_particular_test_or_training_directory = optimised_model_proto.spikes_from_file_data_root + "StimulusSubset" + str(stimulus_subset_index) + "/Testing/"
	# 							# print testing_params.spikes_from_file_particular_test_or_training_directory
	# 							testing_params.spikes_from_file_protocol_number_of_stimuli = len([name for name in os.listdir(testing_params.spikes_from_file_particular_test_or_training_directory + "SubPhase0/") if os.path.isdir(os.path.join(testing_params.spikes_from_file_particular_test_or_training_directory + "SubPhase0/", name))])
	# 							stimulus_manager.stimulus_indices = range(testing_params.spikes_from_file_protocol_number_of_stimuli)


	# 						stimulus_manager.create_stimulus_indices_by_phase_for_test_training_proto(testing_params.lists_of_stimulus_indices_by_phase, stimulus_subset_proto_object.number_of_TESTING_sub_phases_per_training_phase, stimulus_subset_index, False)

	# 						self.proto_helper.save_proto_params_to_file(test_proto_directory_and_name, testing_params)


	# 						shell_file_name = training_phase_directory + "N" + str(self.independent_variable_combination_index) + "_" + str(training_phase_index)
	# 						command_for_shell_file = "./Executables/TestOrTrain " + optimised_model_proto.model_proto_post_optimisation_save_file_name_and_directory + " " + test_proto_directory_and_name + self.command_suffix


	# 						job_id = self.create_shell_script_and_sbatch(shell_file_name, command_for_shell_file, testing_dependency_job_id)
							
	# 						if (training_phase_index == -1):
	# 							training_dependency_job_id = job_id



	# 	combination_complete_file = finished_testing_for_training_phase_file
	# 	return combination_complete_file

