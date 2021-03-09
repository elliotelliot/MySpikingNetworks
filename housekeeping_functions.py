import shutil
import os.path
import time
import sys

##FUNCTION TO MOVE PARAMS TO CURRENT DIRECTORY SO THEY CAN BE IMPORTED
# This should be run any time an old simulation is analysed, to ensure the correct parameters are used
def move_params(simulation_name):

	original = './Outputs/' + simulation_name + '/simulation_params.py'
	target = "./simulation_params.py"
	shutil.copyfile(original, target)


##FUNCTION TO SAVE A COPY OF THE SIM PARAMS FILE
# This should be run every time simulation parameters are updated
#  -- so there is a reference copy of the new parameter file in the relevent folder
def save_params_copy(simulation_name):

	original = "./simulation_params.py"
	target = './Outputs/' + simulation_name + '/simulation_params.py'
	shutil.copyfile(original, target)

##FUNCTION TO SAVE A COPY OF THE SIM CPP FILE
# This should be run every time simulation code (including parameters) is updated
#  -- so there is a reference copy of the new cpp file in the relevent folder
def save_sim_cpp_copy(simulation_name):

	original = "./PolyNetwork.cpp"
	target = './Outputs/' + simulation_name +'/' + simulation_name + "_version.cpp"
	print('Copying ' + original + ' as ' + target + '...\n')
	shutil.copyfile(original, target)

##FUNCTION TO CHECK IF A SIMULATION IS STILL RUNNING
def check_complete(simulation_name, t, t_max):
	
	print("Checking for running simulations...\n\n")
	finished_file_path = "./Outputs/" + simulation_name + "/" + simulation_name + "_finished_check.txt"
	check_value = '0'
	t_total = 0
	while check_value == '0': # Continues to run loop as long as value in 'finished file' is zero or file doesn't exist
		
		if os.path.isfile(finished_file_path) == True: # Checks 'finished file' exists before trying to read it

			finished_file = open(finished_file_path,"r") # Opens 'finished file' in read-only mode
			check_value = finished_file.read() # Reads whole file and assigns value in is as string to 'check_value'
			finished_file.close() #Close file

			if t_total == 0:
				if check_value == '1':

					copy_cpp_file = 0

				else:

					copy_cpp_file = 1

		if t_total > t_max:
			
			sys.exit("ANALYSIS WAS NOT RUN. Simulation failed or took too long. Check outputs and run analysis manually")  

		if check_value == '0':

			time.sleep(t) # Wait t seconds
			t_total = t_total + t # Keep count of approx how long loop has been running (not accurate time)

	return copy_cpp_file   