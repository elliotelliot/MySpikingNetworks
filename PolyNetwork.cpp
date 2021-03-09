#include "Spike/Spike.hpp"
#include <array>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <random>
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h>
#include <chrono>
#include <ctime>    

//sudo scontrol update NodeName=oftserve State=Resume
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*

//////////////////
// DESCRIPTION ///
//////////////////

PolyNetwork is adapted from Niels' BinaryNetwork, but with a single stream. It is set up as follows:
  
  Architecture:
    All layers of neurons (including input) must be the same size
    Any number of excitatory layers supported
    Inhibition can be turned on or off
      -->If inhibition is on, number of inhibitory layers is equal to number of excitatory layers
    Input from a single input layer into the first excitatory layer
    Background input from a single background layer into each excitatory layer

  Input neurons:
    Patterned poisson
    Rates can be set seperately for main and background input

  Main neurons:
    LIF spiking
    Neuron dynamics (resting potential, refractory period, threshold, somatic capacitance, somatic leakage conductance) set for inh and exc neurons seperately 
    ## Note that inhibitory neurons are only inhibitory because of their parameter values! ##
      -->TAKE CARE NOT TO TURN INHIBITORY NEURONS INTO EXCITATORY ONES (OR VICE VERSA)!! !!

  Connectivity:
    Feed-forward, feed-back, and lateral (ex to ex, ex to inh, inh to ex) connections all supported
    Neuron pairs to be connected determined randomly
      -->Probability of a connection between any neuron pair can be set seperately for all connection types
    Supports multiple synapses per connection, each with a different delay (log-normally distributed)
      -->Number of synapses per connection, delay mean and std can be set seperately for each connection type

        //EXPANSION FACTOR: if n_pre > n_post, this implicitly expands the post-synaptic layer to match the presynaptic layer in size
  //if n_pre <= n_post, expansion_factor = 1, and so has no effect
  //example 1: n_pre = 6, n_post = 2, fan_in = 0
    //post_ID = 0; pre_IDs = [0 1 2]; post_ID = 1; pre_IDs = [3 4 5]
  //example 2: n_pre = 8, n_post = 4, fan_in = 2 (* = neurons from fan-in)
    //post_ID = 0; pre_IDs = [0 1 2* 3*]; post_ID = 1; pre_IDs = [0* 1* 2 3 4* 5*]; post_ID = 2; pre_IDs = [2* 3* 4 5 6* 7*]; post_ID = 3; pre_IDs = [4* 5* 6 7]
      //ADJUSTMENT FACTOR: if n_pre < n_post, this implicitly expands the pre-synaptic layer to match the post-synaptic layer in size
  //if n_pre >= n_post, adjustment factor = 1, and so has no effect
  //example 1: n_pre = 2, n_post = 4, fan_in = 0 --> pre layer expanded to [0 0 1 1]
    //post_ID = 0; pre_IDs = 0; post_ID = 1; pre_IDs = 0; post_ID = 2; pre_IDs = 1; post_ID = 3; pre_IDs = 1;
  //example 2: n_pre = 2, n_post = 6, fan_in = 2 --> pre_layer expanded to [0 0 0 1 1 1]
    //post_ID = 0; pre_IDs = [0, 0*, 0*]; post_ID = 1; pre_IDs = [0* 0 0* 1*]; post_ID = 2; pre_IDs = [0* 0* 0 1* 1*];
    //post_ID = 3; pre_IDs = [0*, 0*, 1, 1*, 1*]; post_ID = 4; pre_IDs = [0* 1* 1 1*]; post_ID = 5; pre_IDs = [1* 1* 1];
  
  
  Synapses
    Conductance spiking
    Initial weights uniformally distributed
      -->Initial weight min/max are set seperately for inhibitory and excitatory neurons
    Synaptic dynamics (weight scaling, decay, reversal potential)
    ## Note that inhibitory synapses are only inhibitory because of their parameter values! ##
      -->TAKE CARE NOT TO TURN INHIBITORY SYNAPSES INTO EXCITATORY ONES (OR VICE VERSA)!! !!

  Plasticity
    Custom STDP plasticity rule



*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////
// INITIAL DECLARATIONS ///
///////////////////////////

//DECLARE CONNECTIVIY PARAMETER STRUCTURE
struct connectivity_struct {
    int mult_synapses;
    float delay_min;
    float delay_max;
    float connectivity_param;
    vector<float> fan_in;
    bool learning;
    float weight_scaling;
    int pre_layer_ID;
    int post_layer_ID;
    int n_pre_synaptic_neurons;
    int n_post_synaptic_neurons;
    vector<int> synapses_IDs;
  } input, ff, fb, lat, ex_to_inh, inh_to_ex;

//DECLARE FUNCTIONS
int construct_connectivity(connectivity_struct connectivity, int layer, SpikingModel* model, conductance_spiking_synapse_parameters_struct* synapse_params, CustomSTDPPlasticity* stdp, float timestep);
int pick_pre_synaptic_ID(int post_synaptic_ID, int fan_in, int n_pre_synaptic_neurons, int n_post_synaptic_neurons);
double randnum(double min, double max);

//MAIN FUNCTION
int main (int argc, char *argv[]){

  //RECORD TIME

  auto start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::cout << "Start time: " << std::ctime(&start_time) << endl;


  //CREATE AN INSTANCE OF THE MODEL

  SpikingModel* PolyNetwork = new SpikingModel(); 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /////////////////
  // PARAMETERS ///
  /////////////////
  
  //SET UP SPIKE PARAMETER STRUCTURES
  //only put here what is needed to set parameters for readability/ease of changing them
  lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct(); //create excitatory neuron parameter structure
  lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct(); //create inhibitory neuron parameter structure
  patterned_poisson_input_spiking_neuron_parameters_struct* input_neuron_params = new patterned_poisson_input_spiking_neuron_parameters_struct();
  conductance_spiking_synapse_parameters_struct * input_synapse_params_vec = new conductance_spiking_synapse_parameters_struct();  //create input synapses parameter structure
  conductance_spiking_synapse_parameters_struct * ex_synapse_params_vec = new conductance_spiking_synapse_parameters_struct();  //create excitatory synapses parameter structure
  conductance_spiking_synapse_parameters_struct * inh_synapse_params_vec = new conductance_spiking_synapse_parameters_struct(); //create inhibitory synapses parameter structure
  custom_stdp_plasticity_parameters_struct * ex_stdp_params = new custom_stdp_plasticity_parameters_struct(); //create excitatory stdp parameter structure
  custom_stdp_plasticity_parameters_struct * inh_stdp_params = new custom_stdp_plasticity_parameters_struct(); //create inhibitory stdp parameter structure

  //NAME AND VERSION PRE-AMBLE
  std::string simulation_name = "PolyNetwork_topology"; //output file name ****EDIT NAME HERE
  
  /*
  changes made (from PolyNetowrk_large_sparse_1):
    background input layer removed
    changed "probability of connection" to "n connections"

  changes made (from PolyNetowrk_large_sparse_1):
    4 layers (from 3)
    axonal delay dist: uniform (from normal)
    turned off ex-in and in-ex learning, epoch time 0.4s (from 0.2s)
    1 synapse per connection (from 4)
  
  changes made (from PolyNetwork_large_sparse_2):
    TEST then TRAIN then TEST (not just train then test)
    saving a few different parameters for analysis

  changes made (from PolyNetwork_parameter_tune)
    added option for network topology

  */

  //TRAINING AND TESTING REGIMEN
  // Create a vector called 'regimen' of length equal to total number of epochs in simulation
  // regimen[epoch] = 1 for and epoch with STDP on, and 0 for an epoch with STDP 

  //////////////////////////////
  // parameters here are only used here to set up the training regimen
  // only the vector 'regimen' and int 'total_epochs' are used in the simulation to determine which epochs have STDP on
  // this means you can set up the training and testing regimen however you like
  int pre_train_epochs = 5; //testing epochs prior to training (STDP off)
  int training_epochs = 0; //training epochs (STDP on)
  int post_train_epochs = 0; //testing epochs after training (STDP off)
  /////////////////////////////


  int total_epochs = pre_train_epochs + training_epochs + post_train_epochs;
  vector<int> regimen(total_epochs, 0);
  for (int i=pre_train_epochs; i<pre_train_epochs+training_epochs; i++){
    regimen[i] = 1;
  }

  //TIMESTEP
  float timestep = 0.00002;  // In seconds

  //STIMULUS PARAMETERS
  int input_firing_rate = 50; //Poisson firing rate (Hz) of 'on' input neurons feeding into first layer
  int input_spont_rate = 5; //Poisson firing rate (Hz) of 'off' input neurons feeding into first layer
  float proportion_input_on = 0.2; //Proportion of input neurons 'on'
  float simtime = 0.4f; //This should be long enough to allow any recursive signalling to finish propagating

  //NETWORK ARCHITECTURE
  int n_input_neurons = 16384;
  int n_ex_neurons_per_layer = 4096;
  int n_inh_neurons_per_layer = 1024;
  int n_ex_layers = 4; //number of excitatory layers (not including input layer)
  bool inhibition = 1; //boolean: inhibition on = 1, inhibition off = 0
    //NB: If inhibition is on, number of inhibitory layers is equal to number of excitatory layers

  //EXCITATORY NEURON DYNAMICS
  excitatory_population_params->somatic_capacitance_Cm = 500.0*pow(10, -12);
  excitatory_population_params->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);
  //NB: SPIKE sets membrane time constant = Cm/g0
  excitatory_population_params->resting_potential_v0 = -0.074f;
  excitatory_population_params->threshold_for_action_potential_spike = -0.053f;
  excitatory_population_params->after_spike_reset_potential_vreset = -0.057f;
  excitatory_population_params->absolute_refractory_period = 0.002f; 
  
  //INHIBITORY NEURON DYNAMICS
  inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
  inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
  //NB: SPIKE sets membrane time constant = Cm/g0
  inhibitory_population_params->resting_potential_v0 = -0.082f;
  inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
  inhibitory_population_params->after_spike_reset_potential_vreset = -0.058f;
  inhibitory_population_params->absolute_refractory_period = 0.002f;
  
  //EXCITATORY SYNAPSES
  ex_synapse_params_vec->reversal_potential_Vhat = 0.0*pow(10.0, -3); //v_hat
  ex_synapse_params_vec->decay_term_tau_g = 0.002;
  ex_synapse_params_vec->weight_range[0] = 0.0f;
  ex_synapse_params_vec->weight_range[1] = 1.0f;

  //INHIBITORY SYNAPSES
  inh_synapse_params_vec->reversal_potential_Vhat = -80.0*pow(10.0, -3); //v_hat
  inh_synapse_params_vec->decay_term_tau_g = 0.005;
  inh_synapse_params_vec->weight_range[0] = 0.0f;
  inh_synapse_params_vec->weight_range[1] = 1.0f;
  
  //CONNECTIVITY
  ex_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;
  inh_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;

  //connectivity: input
  input.mult_synapses = 1;
  input.delay_min = 1.0f;
  input.delay_max = 1.0f;
  input.connectivity_param = 30;
  input.fan_in = {0};
  input.learning = 0;
  input.weight_scaling = 0.4*pow(10, -9);

  //connectivity: feed-forward amongst main excitatory layers
  ff.mult_synapses = 1;
  ff.delay_min = 0.1f;
  ff.delay_max = 10.0f;
  ff.connectivity_param = 100;
  ff.fan_in = {8, 12, 16};
  ff.learning = 1;
  ff.weight_scaling = 1.6*pow(10, -9);

  //connectivity: feed-back amongst main excitatory layers
  fb.mult_synapses = 1;
  fb.delay_min = 0.1f;
  fb.delay_max = 10.0f;
  fb.connectivity_param = 10;
  fb.fan_in = {8, 8, 8};
  fb.learning = 1;
  fb.weight_scaling = 1.6*pow(10, -9);

  //connectivity: lateral - excitatory to excitatory
  lat.mult_synapses = 1;
  lat.delay_min = 0.1f;
  lat.delay_max = 10.0f;
  lat.connectivity_param = 10;
  lat.fan_in = {4, 4, 4, 4};
  lat.learning = 1;
  lat.weight_scaling = 1.6*pow(10, -9);

  //connectivity: lateral - excitatory to inhibitory
  ex_to_inh.mult_synapses = 1;
  ex_to_inh.delay_min = 0.1f;
  ex_to_inh.delay_max = 10.0f;
  ex_to_inh.connectivity_param = 30;
  ex_to_inh.fan_in = {0, 0, 0, 0};
  ex_to_inh.learning = 0;
  ex_to_inh.weight_scaling = 80*pow(10, -9);

  //connectivity: lateral - inhibitory to excitatory
  inh_to_ex.mult_synapses = 1;
  inh_to_ex.delay_min = 0.1f;
  inh_to_ex.delay_max = 10.0f;
  inh_to_ex.connectivity_param = 30;
  inh_to_ex.fan_in = {8, 8, 8, 8};
  inh_to_ex.learning = 0;
  inh_to_ex.weight_scaling = 40*pow(10, -9);

  /////////////////////////


  //PLASTICITY

  //excitatory
  ex_stdp_params->a_plus = 0.5f; //set to the mean of the excitatory weight distribution
  ex_stdp_params->a_minus = 0.5f;
  ex_stdp_params->weight_dependence_power_ltd = 0.0f; //by setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  ex_stdp_params->w_max = 1; //sets the maximum weight that can be *learned* (hard border)
  ex_stdp_params->tau_plus = 0.05f;
  ex_stdp_params->tau_minus = 0.05f;
  ex_stdp_params->learning_rate = 0.1f;
  ex_stdp_params->a_star = 0; //ex_stdp_params->a_plus * ex_stdp_params->tau_minus * Inhib_STDP_PARAMS->targetrate;

  //inhibitory
  inh_stdp_params->a_plus = 0.5f; //set to the mean of the inhibitory weight distribution
  inh_stdp_params->a_minus = 0.5f;
  inh_stdp_params->weight_dependence_power_ltd = 0.0f; //by setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  inh_stdp_params->w_max = 1; //sets the maximum weight that can be *learned* (hard border)
  inh_stdp_params->tau_plus = 0.05f;
  inh_stdp_params->tau_minus = 0.05f;
  inh_stdp_params->learning_rate = 0.1f;
  inh_stdp_params->a_star = 0; //ex_stdp_params->a_plus * ex_stdp_params->tau_minus * Inhib_STDP_PARAMS->targetrate;
  

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////
  // CREATE FILE AND DIRECTORIES ///
  //////////////////////////////////

  //CREATE OUPUT DIRECTORIES

  std::string output_dir = "./Outputs/" + simulation_name;
  const char* output_dir_c = output_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function (linux)
  int check = mkdir(output_dir_c, 0777); //make file
  /*if (check != 0){ //check file could be made - i.e. it doesn't already exist!
    std::cout << "WARNING: a simulation of this name has already been executed. Edit output directory path before continuing to avoid overwriting data.\n\nProgramme exiting.";
    return 0;
  }*/

  std::string input_neurons_dir = output_dir + "/Input_neurons";
  const char* input_neurons_dir_c = input_neurons_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function
  mkdir(input_neurons_dir_c, 0777); //make file

  std::string main_neurons_dir = output_dir + "/Main_neurons";
  const char* main_neurons_dir_c = main_neurons_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function
  mkdir(main_neurons_dir_c, 0777); //make file

  std::string weight_evolution_dir = output_dir + "/Weight_evolution";
  const char* weight_evolution_dir_c = weight_evolution_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function
  mkdir(weight_evolution_dir_c, 0777); //make file

  std::string figures_dir = output_dir + "/Figures";
  const char* figures_dir_c = figures_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function
  mkdir(figures_dir_c, 0777); //make file

  //CREATE 'FINISHED' FILE
  //This is a text file that contains a '0' while a simulation is running
  //When the simulation is finished, this file will be overwritten, with the new file containing a '1'
  //Any check on the value within the file should first check if the file exists, and then check the value

  std::string finished_file = output_dir + "/" + simulation_name + "_finished_check.txt";
  std::ofstream outfile (finished_file); //overwrites or creates 'finished file'
  outfile << 0; //writes '0' in 'finished file'
  outfile.close(); //closes



  //////////////////////////
  // SET UP NEURON GROUPS ///
  ///////////////////////////

  std::cout << "\n\n.......\nSetting up neuron groups...\n.......\n\n";

  //set the timestep and neurons first
  PolyNetwork->SetTimestep(timestep);
  LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons(); //choose neuron type
  PolyNetwork->spiking_neurons = lif_spiking_neurons; //assign neuron type to model

  // SETTING UP INPUT NEURONS

  //choose input neuron type
  PatternedPoissonInputSpikingNeurons* patterned_poisson_input_neurons = new PatternedPoissonInputSpikingNeurons();
  PolyNetwork->input_spiking_neurons = patterned_poisson_input_neurons;

  //finish input parameter structure
  input_neuron_params->group_shape[0] = 1;    // x-dimension of the input neuron layer
  input_neuron_params->group_shape[1] = n_input_neurons; // y-dimension of the input neuron layer

  //create a group of input neurons. This function returns the ID of the input neuron group
  int input_layer_ID = PolyNetwork->AddInputNeuronGroup(input_neuron_params);
  std::cout << "Input layer ID is " << input_layer_ID << "\n";
  
  // SETTING UP MAIN GROUPS

  //finish excitatory parameter structure
  excitatory_population_params->group_shape[0] = 1;
  excitatory_population_params->group_shape[1] = n_ex_neurons_per_layer;
  
  //initialise vector to store layer IDs
  vector<int> ex_layer_IDs(n_ex_layers, 0);
    
  //iteratively add each excitatory layer to network
  for (int i=0; i<n_ex_layers; i++){
    ex_layer_IDs[i] = PolyNetwork->AddNeuronGroup(excitatory_population_params);
    std::cout << "New excitatory layer ID is " << ex_layer_IDs[i] << "\n";
  }

  //initialise vector to store layer IDs
  vector<int> inh_layer_IDs(n_ex_layers, 0);

  //if there are inhbitory layers
  if (inhibition == 1){

    //finish inhibitory parameter structure
    inhibitory_population_params->group_shape[0] = 1;
    inhibitory_population_params->group_shape[1] = n_inh_neurons_per_layer;

    //iteratively add each inhibitory layer to network
    for (int i=0; i<n_ex_layers; i++){
      inh_layer_IDs[i] = PolyNetwork->AddNeuronGroup(inhibitory_population_params);
      std::cout << "New inhibitory layer ID is " << inh_layer_IDs[i] << "\n";
    }
  }



 
  //////////////////////
  // SET UP SYNAPSES ///
  //////////////////////

  std::cout << "\n\n.......\nSetting up synapses...\n.......\n\n";

  //choose synapse type
  ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
  PolyNetwork->spiking_synapses = conductance_spiking_synapses;

  //assign plasticity rules
  CustomSTDPPlasticity * excitatory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) ex_stdp_params);  
  PolyNetwork->AddPlasticityRule(excitatory_stdp);
  CustomSTDPPlasticity * inhibitory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) inh_stdp_params);  
  PolyNetwork->AddPlasticityRule(inhibitory_stdp);
  
  //set arbitrary values for delay range - needed by spike, but amended later
  ex_synapse_params_vec->delay_range[0] = 1*timestep; ex_synapse_params_vec->delay_range[1] = 10*timestep;
  inh_synapse_params_vec->delay_range[0] = 1*timestep; inh_synapse_params_vec->delay_range[1] = 10*timestep;

  //initialise vector which contains list of synapse group IDs for which weights will be saved during training
  //only saving weights for synapse groups where learning is on
  vector<int> weights_to_save;
  vector<std::string> group_names;

  //input to first excitatory layer
  std::cout << "Input synapses:\n";
  if (input.fan_in.size() != 1){
    std::cout << "Error: fan in radii vector size for input synapses must equal 1";
    return(0);
  }
  input.n_pre_synaptic_neurons = n_input_neurons;
  input.n_post_synaptic_neurons = n_ex_neurons_per_layer;
  input.pre_layer_ID = input_layer_ID;
  input.post_layer_ID = ex_layer_IDs[0];
  input.synapses_IDs.push_back(construct_connectivity(input, 0, PolyNetwork, ex_synapse_params_vec, excitatory_stdp, timestep));
  if (input.learning){
    weights_to_save.push_back(input.synapses_IDs[0]);
    group_names.push_back("Input");
  }
  
  
  //feed-forward
  std::cout << "Feed-forward synapses:\n";
  if (ff.fan_in.size() != n_ex_layers-1){
    std::cout << "Error: fan in radii vector size for feed-forward synapses must equal number of layers minus 1";
    return(0);
  }
  ff.n_pre_synaptic_neurons = n_ex_neurons_per_layer;
  ff.n_post_synaptic_neurons = n_ex_neurons_per_layer;
  for (int i=0; i<(n_ex_layers-1); i++){
    ff.pre_layer_ID = ex_layer_IDs[i];
    ff.post_layer_ID = ex_layer_IDs[(i+1)];
    ff.synapses_IDs.push_back(construct_connectivity(ff, i, PolyNetwork, ex_synapse_params_vec, excitatory_stdp, timestep));
    if (ff.learning){
      weights_to_save.push_back(ff.synapses_IDs[i]);
      group_names.push_back("Feed-forward_" + std::to_string(i));
    }
  }

  //feed-back
  std::cout << "Feed-back synapses:\n";
  if (fb.fan_in.size() != n_ex_layers-1){
    std::cout << "Error: fan in radii vector size for feed-back synapses must equal number of layers minus 1";
    return(0);
  }
  fb.n_pre_synaptic_neurons = n_ex_neurons_per_layer;
  fb.n_post_synaptic_neurons = n_ex_neurons_per_layer;
  for (int i=0; i<(n_ex_layers-1); i++){
    fb.pre_layer_ID = ex_layer_IDs[i+1];
    fb.post_layer_ID = ex_layer_IDs[i];
    fb.synapses_IDs.push_back(construct_connectivity(fb, i, PolyNetwork, ex_synapse_params_vec, excitatory_stdp, timestep));
    if (fb.learning){
      weights_to_save.push_back(fb.synapses_IDs[i]);
      group_names.push_back("Feed-back_" + std::to_string(i));
    }
  }

  //lateral - excitatory to excitatory
  std::cout << "Lateral excitatory to excitatory synapses:\n";
  if (lat.fan_in.size() != n_ex_layers){
    std::cout << "Error: fan in radii vector size for lateral E->E synapses must equal number of layers";
    return(0);
  }
  lat.n_pre_synaptic_neurons = n_ex_neurons_per_layer;
  lat.n_post_synaptic_neurons = n_ex_neurons_per_layer;
  for (int i=0; i<n_ex_layers; i++){
    lat.pre_layer_ID = ex_layer_IDs[i];
    lat.post_layer_ID = ex_layer_IDs[i];
    lat.synapses_IDs.push_back(construct_connectivity(lat, i, PolyNetwork, ex_synapse_params_vec, excitatory_stdp, timestep));
    if (lat.learning){
      weights_to_save.push_back(lat.synapses_IDs[i]);
      group_names.push_back("Lateral_" + std::to_string(i));
    }
  }

  if (inhibition == 1){ //if there is inhibition

    //lateral - excitatory to inhibitory
    std::cout << "Lateral excitatory to inhibitory synapses:\n";
    if (ex_to_inh.fan_in.size() != n_ex_layers){
      std::cout << "Error: fan in radii vector size for lateral E->I synapses must equal number of layers";
      return(0);
    }
    ex_to_inh.n_pre_synaptic_neurons = n_ex_neurons_per_layer;
    ex_to_inh.n_post_synaptic_neurons = n_inh_neurons_per_layer;
    for (int i=0; i<n_ex_layers; i++){
      ex_to_inh.pre_layer_ID = ex_layer_IDs[i];
      ex_to_inh.post_layer_ID = inh_layer_IDs[i];
      ex_to_inh.synapses_IDs.push_back(construct_connectivity(ex_to_inh, i, PolyNetwork, ex_synapse_params_vec, excitatory_stdp, timestep));
      if (ex_to_inh.learning){
        weights_to_save.push_back(ex_to_inh.synapses_IDs[i]);
        group_names.push_back("Ex-to-inh_" + std::to_string(i));
      }
    }
    
    //lateral - inhibitory to excitatory
    std::cout << "Lateral inhibitory to excitatory synapses:\n";
    if (inh_to_ex.fan_in.size() != n_ex_layers){
      std::cout << "Error: fan in radii vector size for lateral I->E synapses must equal number of layers";
      return(0);
    }
    inh_to_ex.n_pre_synaptic_neurons = n_inh_neurons_per_layer;
    inh_to_ex.n_post_synaptic_neurons = n_ex_neurons_per_layer;
    for (int i=0; i<n_ex_layers; i++){
      inh_to_ex.pre_layer_ID = inh_layer_IDs[i];
      inh_to_ex.post_layer_ID = ex_layer_IDs[i];
      inh_to_ex.synapses_IDs.push_back(construct_connectivity(inh_to_ex, i, PolyNetwork, inh_synapse_params_vec, excitatory_stdp, timestep));
      if (inh_to_ex.learning){
        weights_to_save.push_back(inh_to_ex.synapses_IDs[i]);
        group_names.push_back("Inh-to-ex_" + std::to_string(i));
      } 
    }
  }

  //prepare to save weights group by group
  int n_weight_groups_to_save = weights_to_save.size();
 

  ////////////////////////
  // ACTIVITY MONITORS ///
  ////////////////////////

  std::cout << "\n\n.......\nAssigning activity monitors to the network...\n.......\n\n";

  //input spike monitor
  SpikingActivityMonitor* spike_monitor_input = new SpikingActivityMonitor(patterned_poisson_input_neurons);
  PolyNetwork->AddActivityMonitor(spike_monitor_input);
  std::cout << "Input spike monitor added\n";

  //main spike monitor
  SpikingActivityMonitor* spike_monitor_main = new SpikingActivityMonitor(lif_spiking_neurons);
  PolyNetwork->AddActivityMonitor(spike_monitor_main);
  std::cout << "Main spike monitor added\n";
  

  /////////////////////
  // SET UP STIMULI ///
  /////////////////////

  std::cout << "\n\n.......\nAssigning stimuli to the network...\n.......\n\n";
  
  //create vector of rates for 'on' input neurons
  int number_on = proportion_input_on*n_input_neurons;
  std::vector<float> input_rates(number_on, input_firing_rate);

  //append 'off' neurons
  int number_off = n_input_neurons - number_on;
  for (int i=0; i < number_off; i++){
    input_rates.push_back(input_spont_rate);
  }

  
  //print input stimulus
  std::cout << "Input:\n";
  std::cout << std::to_string(number_on) << " neurons firing at " << std::to_string(input_firing_rate) << "Hz\n";
  std::cout << std::to_string(number_off) << " neurons firing at " << std::to_string(input_spont_rate) << "Hz\n";
  

  //assign rates to stimulus
  int stimulus = patterned_poisson_input_neurons->add_stimulus(input_rates);
  std::cout << "\n\nNew stimulus ID is " << std::to_string(stimulus) << "\n";

  PolyNetwork->finalise_model();

  std::cout << "\n\n.......\nModel finalised and ready for simulating...\n.......\n\n";


  /////////////////////
  // RUN SIMULATION ///
  /////////////////////

  PolyNetwork->spiking_synapses->save_weights_as_binary(output_dir, "Initial_");

  PolyNetwork->spiking_synapses->save_weights_as_txt(output_dir, "Initial_");

  for (int jj = 0; jj < n_weight_groups_to_save; ++jj){
      PolyNetwork->spiking_synapses->save_weights_as_binary(weight_evolution_dir, "Initial_" + group_names[jj] + "_", weights_to_save[jj]);
    }

  // Loop through a certain number of epochs of presentation
  for (int ii = 0; ii < total_epochs; ++ii) {

    patterned_poisson_input_neurons->select_stimulus(stimulus);

    PolyNetwork->run(simtime, regimen[ii]); //the second argument is a boolean determining if STDP is on or off

    if (regimen[ii]){
      for (int jj = 0; jj < n_weight_groups_to_save; ++jj){
        PolyNetwork->spiking_synapses->save_weights_as_binary(weight_evolution_dir, "Epoch_" + std::to_string(ii) + "_" + group_names[jj] + "_", weights_to_save[jj]);
      }
    }
    
    spike_monitor_main->save_spikes_as_binary(main_neurons_dir, "Epoch_" + std::to_string(ii) + "_");
    spike_monitor_main->save_spikes_as_txt(main_neurons_dir, "Epoch_" + std::to_string(ii) + "_");
    spike_monitor_input->save_spikes_as_binary(input_neurons_dir, "Epoch_" + std::to_string(ii) + "_");
    spike_monitor_main->reset_state(); //Dumps all recorded spikes
    spike_monitor_input->reset_state(); //Dumps all recorded spikes
    PolyNetwork->reset_time(); //Resets the internal clock to 0
  }

  PolyNetwork->spiking_synapses->save_weights_as_binary(output_dir, "Final_");



  ////////////////////////////
  // SAVE PYTHON VARIABLES ///
  ////////////////////////////

  std::cout << "\n\n.......\nWriting parameters to file\n.......\n\n";

  std::string time_string = std::string(ctime(&start_time)); //converts start time to string format, ending with a \n
  time_string.pop_back(); //removes \n
  std::string simulation_params_file_path = output_dir + "/simulation_params.py"; //name for parameters file
  std::ofstream params_file; //define pointer to this file
  params_file.open(simulation_params_file_path, ios::trunc); //opens file in overwrite mode
  params_file << "simulation_name = '" << simulation_name << "'" << std::endl;
  params_file << "sim_output_folder = './Outputs/" << simulation_name << "/'" << std::endl;
  params_file << "figure_folder = './Outputs/" << simulation_name << "/Figures/'" << std::endl;
  params_file << "n_epochs = " << total_epochs << std::endl;
  params_file << "regimen = [" << regimen[0];
  for (int i = 1; i<total_epochs; i++){
    params_file << ", " << regimen[i];
  }
  params_file << "]" << std::endl;
  params_file << "timestep = " << timestep << std::endl;
  params_file << "n_ex_layers = " << n_ex_layers << std::endl;
  params_file << "inhibition_bool = " << inhibition << std::endl;
  params_file << "n_input_neurons = " << n_input_neurons << std::endl;
  params_file << "n_ex_neurons_per_layer = " << n_ex_neurons_per_layer << std::endl;
  params_file << "n_inh_neurons_per_layer = " << n_inh_neurons_per_layer << std::endl;
  params_file << "simtime = " << simtime << std::endl;
  params_file << "group_names = [\"" << group_names[0] << "\"";
  for (int i = 1; i<n_weight_groups_to_save; i++){
    params_file << ",\"" << group_names[i] << "\"";
  }
  params_file << "]" << std::endl;
  params_file << "sim_start_time = '" << time_string << "'";
  params_file.close(); //closes file

  /////////////
  // FINISH ///
  /////////////

  std::cout << "\n\n.......\nWriting finished check to file\n.......\n\n";

  outfile.open(finished_file, ios::trunc); //opens file in overwrite mode
  outfile << 1; //overwrites '0' with '1'
  outfile.close(); //closes

  std::cout << "\n\n.......\nFinished Simulation\n.......\n\n";

  auto end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::cout << "End time: " << std::ctime(&end_time) << endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//FUNCTION TO RANDOMLY PICK A PRE-SYNAPTIC NEURON FROM WITHIN FAN-IN RADIUS

int pick_pre_synaptic_ID(int post_synaptic_ID, int fan_in, int n_pre_synaptic_neurons, int n_post_synaptic_neurons)
{

  //CALCULATE EXPANSION FACTOR (for when n_pre_synaptic_neurons < n_post_synaptic neurons -- see top for explanation)
  int expansion_factor = ceil( (float) n_pre_synaptic_neurons/ (float)n_post_synaptic_neurons);
  
  //CALCULATE ADJUSTMENT FACTOR (for when n_pre_synaptic_neurons > n_post_synaptic neurons -- see top for explanation)
  float adjustment_factor;
  if (n_post_synaptic_neurons > n_pre_synaptic_neurons){
    adjustment_factor = (float)n_pre_synaptic_neurons / (float)n_post_synaptic_neurons;
  }
  else{
    adjustment_factor = 1.0;
  }
  
  //CALCULATE START AND END OF RANGE OF PRE-SYNAPTIC NEURONS TO CHOOSE FROM
  int start_ID = expansion_factor * post_synaptic_ID - fan_in;
  int end_ID = expansion_factor * post_synaptic_ID + expansion_factor - 1 + fan_in;
  
  //ADJUST FOR EDGE NEURONS
  if (start_ID < 0){
    start_ID = 0; //chosen pre-synaptic ID can't be less than 0
  }
  if (end_ID > (n_pre_synaptic_neurons / adjustment_factor - 1)){
    end_ID = n_pre_synaptic_neurons / adjustment_factor - 1; //chosen pre-synaptic ID can't be greater than final pre-synaptic ID
  }

  //PICK RANDOM NEURON ID IN RANGE
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); //choose seed from time
  std::default_random_engine generator(seed); //seed random number generator
  std::uniform_int_distribution<int> distribution(start_ID, end_ID); //create distribution
  int neuron_ID = floor(distribution(generator) * adjustment_factor); //pick number from distribution and adjust
  return(neuron_ID); //return chosen presynaptic ID

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//FUNCTION TO PICK A RANDOM NUMBER FROM RANGE

double randnum (double min, double max)
{
  //prepare random number generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  //create distribution
  std::uniform_real_distribution<double> distribution(min, max);

  //pick number from distribution and return
  return distribution(generator);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//FUNCTION TO CONSTRUCT CONNECTIVITY

int construct_connectivity(connectivity_struct connectivity, int layer, SpikingModel* model, conductance_spiking_synapse_parameters_struct* synapse_params, CustomSTDPPlasticity * stdp, float timestep)
{

  //CLEAR VECTORS
  synapse_params->pairwise_connect_presynaptic.clear();
  synapse_params->pairwise_connect_postsynaptic.clear();
  synapse_params->plasticity_vec.clear();
  synapse_params->pairwise_connect_delay.clear();

  //SET WEIGHT SCALING CONSTANT
  synapse_params->weight_scaling_constant = connectivity.weight_scaling;

  //TURN ON LEARNING
  if (connectivity.learning){
    synapse_params->plasticity_vec.push_back(stdp);
  }

  //CHOOSE PAIRWISE CONNECTIVITY
  int fan_in = connectivity.fan_in[layer];
  int pre_synaptic_ID; //declare
  for (int post_synaptic_ID = 0; post_synaptic_ID < connectivity.n_post_synaptic_neurons; post_synaptic_ID++){ //cycle through post-synaptic neurons
    for (int i=0; i < connectivity.connectivity_param; i++){ //run the two below lines x times, where x is the number of afferent connections per post-synaptic neuron
      synapse_params->pairwise_connect_postsynaptic.push_back(post_synaptic_ID); //put the relevant postynaptic ID into the connectivity vector
      pre_synaptic_ID = pick_pre_synaptic_ID(post_synaptic_ID, fan_in, connectivity.n_pre_synaptic_neurons, connectivity.n_post_synaptic_neurons); //pick pre-synaptic ID
      synapse_params->pairwise_connect_presynaptic.push_back(pre_synaptic_ID); //put the randomly selected presynaptic ID into the connectivity vector
    }
  }

  //RANDOMISE DELAYS
  int n_synapses = synapse_params->pairwise_connect_presynaptic.size();
  for (int j = 0; j <n_synapses; j++){
    synapse_params->pairwise_connect_delay.push_back(randnum(connectivity.delay_min, connectivity.delay_max)/1000 + timestep); //distribution is in seconds, so must be scaled to ms. a small value added to prevent floating point errors in spike
  }

  //PRINT SANITY CHECK
  std::cout << "Layer ID " << std::to_string(connectivity.pre_layer_ID) << " is sending " << std::to_string(n_synapses) << " synapses to layer ID " << std::to_string(connectivity.post_layer_ID) << "\n";


  //ADD SYNAPSE GROUP AND RETURN ID
  int synapse_group_ID = model->AddSynapseGroup(connectivity.pre_layer_ID, connectivity.post_layer_ID, synapse_params);
  return synapse_group_ID;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

