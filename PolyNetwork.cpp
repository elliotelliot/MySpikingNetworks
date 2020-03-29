#include "Spike/Spike.hpp"
//#include "UtilityFunctionsLeadholm.hpp"
#include <array>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <random>
#include <chrono>
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 

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
    float delay_mean;
    float delay_std;
    float prob_connection;
    bool learning;
  } input, background, ff, fb, lat, ex_to_inh, inh_to_ex;

//DECLARE FUNCTION TO CONSTRUCT CONNECTIVITY
void construct_connectivity(SpikingModel* model, connectivity_struct connectivity, int pre_layer_ID, int post_layer_ID, conductance_spiking_synapse_parameters_struct* synapse_params, CustomSTDPPlasticity* stdp, float weight_min, float weight_max, int n_neurons_per_layer, float timestep);


//MAIN FUNCTION
int main (int argc, char *argv[]){

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
  conductance_spiking_synapse_parameters_struct * ex_synapse_params_vec = new conductance_spiking_synapse_parameters_struct();  //create excitatory synapses parameter structure
  conductance_spiking_synapse_parameters_struct * inh_synapse_params_vec = new conductance_spiking_synapse_parameters_struct(); //create inhibitory synapses parameter structure
  custom_stdp_plasticity_parameters_struct * ex_stdp_params = new custom_stdp_plasticity_parameters_struct(); //create excitatory stdp parameter structure
  custom_stdp_plasticity_parameters_struct * inh_stdp_params = new custom_stdp_plasticity_parameters_struct(); //create inhibitory stdp parameter structure

  //SIMULATION
  std::string simulation_name = "PolyNetwork_test"; //output file name ****EDIT NAME HERE
  int training_epochs = 10; // Number of epochs to have STDP active
  int display_epochs = 10; // Number of epochs where the each stimulus is presented with STDP inactive
  float timestep = 0.0001;  // In seconds
  int input_firing_rate = 50; //Poisson firing rate (Hz) of input neurons feeding into first layer
  int background_firing_rate = 500; // Poisson firing rate (Hz) of noisy neurons feeding into all layers, preventing dead neurons
  float simtime = 0.2f; //This should be long enough to allow any recursive signalling to finish propagating

  //NETWORK ARCHITECTURE
  int x_dim = 5; int y_dim = 5; //dimensions of each layer - same for all layers
  int n_ex_layers = 3; //number of excitatory layers (not including input layer)
  bool inhibition = 1; //boolean: inhibition on = 0, inhibition off = 1
    //NB: If inhibition is on, number of inhibitory layers is equal to number of excitatory layers

  //EXCITATORY NEURON DYNAMICS
  excitatory_population_params->resting_potential_v0 = -0.06f;
  excitatory_population_params->absolute_refractory_period = 0.002f;
  excitatory_population_params->threshold_for_action_potential_spike = -0.05f;
  excitatory_population_params->somatic_capacitance_Cm = 200.0*pow(10, -12);
  excitatory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
  
  //INHIBITORY NEURON DYNAMICS
  inhibitory_population_params->resting_potential_v0 = -0.082f;
  inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
  inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
  inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);

  //CONNECTIVITY

  //input to first layer
  input.mult_synapses = 4;
  input.delay_mean = 3.4f;
  input.delay_std = 2.3f;
  input.prob_connection = 1;
  input.learning = 1;

  //background to all layers
  background.mult_synapses = 4;
  background.delay_mean = 3.4f;
  background.delay_std = 2.3f;
  background.prob_connection = 1;
  background.learning = 0;

  //feed-forward amongst main excitatory layers
  ff.mult_synapses = 4;
  ff.delay_mean = 3.4f;
  ff.delay_std = 2.3f;
  ff.prob_connection = 1;
  ff.learning = 1;

  //feed-back amongst main excitatory layers
  fb.mult_synapses = 4;
  fb.delay_mean = 3.4f;
  fb.delay_std = 2.3f;
  fb.prob_connection = 1;
  fb.learning = 1;

  //lateral - excitatory to excitatory
  lat.mult_synapses = 4;
  lat.delay_mean = 3.4f;
  lat.delay_std = 2.3f;
  lat.prob_connection = 1;
  lat.learning = 1;

  //lateral - excitatory to inhibitory
  ex_to_inh.mult_synapses = 4;
  ex_to_inh.delay_mean = 3.4f;
  ex_to_inh.delay_std = 2.3f;
  ex_to_inh.prob_connection = 1;
  ex_to_inh.learning = 1;

  //lateral - inhibitory to excitatory
  inh_to_ex.mult_synapses = 4;
  inh_to_ex.delay_mean = 3.4f;
  inh_to_ex.delay_std = 2.3f;
  inh_to_ex.prob_connection = 1;
  inh_to_ex.learning = 1;

  
  //EXCITATORY SYNAPTIC DYNAMICS
  float ex_weight_min = 0.005; float ex_weight_max = 0.015; //initial weights min/max - follows uniform dist
  ex_synapse_params_vec->decay_term_tau_g = 0.0017f;  //conductance parameter (seconds)
  ex_synapse_params_vec->reversal_potential_Vhat = 0.0*pow(10.0, -3); //v_hat
  ex_synapse_params_vec->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;

  //INHIBITORY SYNAPTIC DYNAMICS
  float inh_weight_min = ex_weight_min*5; float inh_weight_max = ex_weight_max*5; //initial weights min/max - follows uniform dist
  inh_synapse_params_vec->decay_term_tau_g = 0.0017f; //conductance parameter (seconds)
  inh_synapse_params_vec->reversal_potential_Vhat = -80.0*pow(10.0, -3); //v_hat
  inh_synapse_params_vec->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0*5.0; //inhibitory weights scaled to be greater than excitatory weights

  //EXCITATORY PLASTICITY
  ex_stdp_params->a_plus = 1.0f; //set to the mean of the excitatory weight distribution
  ex_stdp_params->a_minus = 1.0f;
  ex_stdp_params->weight_dependence_power_ltd = 0.0f; //by setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  ex_stdp_params->w_max = 0.03; //sets the maximum weight that can be *learned* (hard border)
  ex_stdp_params->tau_plus = 0.01f;
  ex_stdp_params->tau_minus = 0.01f;
  ex_stdp_params->learning_rate = 0.001f;
  ex_stdp_params->a_star = 0; //ex_stdp_params->a_plus * ex_stdp_params->tau_minus * Inhib_STDP_PARAMS->targetrate;

  //INHIBITORY PLASTICITY
  inh_stdp_params->a_plus = 1.0f; //set to the mean of the inhibitory weight distribution
  inh_stdp_params->a_minus = 1.0f;
  inh_stdp_params->weight_dependence_power_ltd = 0.0f; //by setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  inh_stdp_params->w_max = 0.03; //sets the maximum weight that can be *learned* (hard border)
  inh_stdp_params->tau_plus = 0.01f;
  inh_stdp_params->tau_minus = 0.01f;
  inh_stdp_params->learning_rate = 0.001f;
  inh_stdp_params->a_star = 0; //ex_stdp_params->a_plus * ex_stdp_params->tau_minus * Inhib_STDP_PARAMS->targetrate;
  

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////
  // SAVE PARAMETERS ///
  //////////////////////


  //CREATE OUTPUT FILES

  std::string output_dir = "./Outputs/" + simulation_name;
  const char* output_dir_c = output_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function (linux)
  int check = mkdir(output_dir_c, 0777); //make file
  if (check != 0){ //check file could be made - i.e. it doesn't already exist!
    std::cout << "WARNING: a simulation of this name has already been executed. Edit output directory path before continuing to avoid overwriting data.\n\nProgramme exiting.";
    return 0;
  }

  std::string training_data_dir = output_dir + "/Training_data";
  const char* training_data_dir_c = training_data_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function
  mkdir(training_data_dir_c, 0777); //make file

  std::string testing_data_dir = output_dir + "/Testing_data";
  const char* testing_data_dir_c = testing_data_dir.c_str(); //convert dir to c string - needed to be accepted by mkdir function
  mkdir(testing_data_dir_c, 0777); //make file

  //SAVE PARAMETERS
  int n_inh_layers = n_ex_layers*inhibition;
  std::string analysis_dir = "./Analysis";
  std::string simulation_params_file = analysis_dir + "/simulation_params.py";
  std::ofstream outfile (simulation_params_file);
  outfile << "simulation_name = '" << simulation_name << "'" << std::endl;
  outfile << "training_epochs = " << training_epochs << std::endl;
  outfile << "display_epochs = " << display_epochs << std::endl;
  outfile << "timestep = " << timestep << std::endl;
  outfile << "n_ex_layers = " << n_ex_layers << std::endl;
  outfile << "n_inh_layers = " << n_inh_layers << std::endl;
  outfile << "n_neurons_per_layer = " << x_dim*y_dim << std::endl;
  outfile << "simtime = " << simtime << std::endl;
  outfile.close();


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
  input_neuron_params->group_shape[0] = x_dim;    // x-dimension of the input neuron layer
  input_neuron_params->group_shape[1] = y_dim;    // y-dimension of the input neuron layer

  //create a group of input neurons. This function returns the ID of the input neuron group
  int input_layer_ID = PolyNetwork->AddInputNeuronGroup(input_neuron_params);
  std::cout << "New input layer ID is " << input_layer_ID << "\n";

  //create a group of background input neurons
  int background_layer_ID = PolyNetwork->AddInputNeuronGroup(input_neuron_params);
  std::cout << "New background input layer ID is " << background_layer_ID << "\n";

  // SETTING UP MAIN GROUPS

  //finish excitatory parameter structure
  excitatory_population_params->group_shape[0] = x_dim;
  excitatory_population_params->group_shape[1] = y_dim;
  
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
    inhibitory_population_params->group_shape[0] = x_dim;
    inhibitory_population_params->group_shape[1] = y_dim;

    //iteratively add each inhibitory layer to network
    for (int i=0; i<n_ex_layers; i++){
      inh_layer_IDs[i] = PolyNetwork->AddNeuronGroup(inhibitory_population_params);
      std::cout << "New inhibitory layer ID is " << inh_layer_IDs[i] << "\n";
    }
  }



 
  //////////////////////
  // SET UP SYNAPSES ///
  //////////////////////

  std::cout << "\n\n.......\nSetting up excitatory synapses...\n.......\n\n";

  //choose synapse type
  ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
  PolyNetwork->spiking_synapses = conductance_spiking_synapses;

  //finish excitatory parameter structure
  ex_synapse_params_vec->delay_range[0] = 10.0*timestep;
  ex_synapse_params_vec->delay_range[1] = 10.0*timestep; //NB that as the delays will be set later, these values are arbitrary, albeit required by Spike
  ex_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;

  //assign plasticity rule
  CustomSTDPPlasticity * excitatory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) ex_stdp_params);  
  PolyNetwork->AddPlasticityRule(excitatory_stdp);

  //calculate number of neurons per layer
  int n_neurons_per_layer = x_dim*y_dim;
  
  //input to first layer
  int pre_layer_ID = input_layer_ID;
  int post_layer_ID = ex_layer_IDs[0];
  construct_connectivity(PolyNetwork, input, pre_layer_ID, post_layer_ID, ex_synapse_params_vec, excitatory_stdp, ex_weight_min, ex_weight_max, n_neurons_per_layer, timestep);

  //background to all layers
  for (int i=0; i<n_ex_layers; i++){
    pre_layer_ID = background_layer_ID;
    post_layer_ID = ex_layer_IDs[i];
    construct_connectivity(PolyNetwork, background, pre_layer_ID, post_layer_ID, ex_synapse_params_vec, excitatory_stdp, ex_weight_min, ex_weight_max, n_neurons_per_layer, timestep);

  }

  //feed-forward
  for (int i=0; i<(n_ex_layers-1); i++){
    pre_layer_ID = ex_layer_IDs[i];
    post_layer_ID = ex_layer_IDs[(i+1)];
    construct_connectivity(PolyNetwork, ff, pre_layer_ID, post_layer_ID, ex_synapse_params_vec, excitatory_stdp, ex_weight_min, ex_weight_max, n_neurons_per_layer, timestep);
  }

  //feed-back
  for (int i=0; i<(n_ex_layers-1); i++){
    pre_layer_ID = ex_layer_IDs[i+1];
    post_layer_ID = ex_layer_IDs[i];
    construct_connectivity(PolyNetwork, fb, pre_layer_ID, post_layer_ID, ex_synapse_params_vec, excitatory_stdp, ex_weight_min, ex_weight_max, n_neurons_per_layer, timestep);
  }

  //lateral - excitatory to excitatory
  for (int i=0; i<n_ex_layers; i++){
    pre_layer_ID = ex_layer_IDs[i];
    post_layer_ID = ex_layer_IDs[i];
    construct_connectivity(PolyNetwork, lat, pre_layer_ID, post_layer_ID, ex_synapse_params_vec, excitatory_stdp, ex_weight_min, ex_weight_max, n_neurons_per_layer, timestep);
  }

  
  if (inhibition == 1){ //if there is inhibition

    //lateral - excitatory to inhibitory
    for (int i=0; i<n_ex_layers; i++){
      pre_layer_ID = ex_layer_IDs[i];
      post_layer_ID = inh_layer_IDs[i];
      construct_connectivity(PolyNetwork, ex_to_inh, pre_layer_ID, post_layer_ID, ex_synapse_params_vec, excitatory_stdp, ex_weight_min, ex_weight_max, n_neurons_per_layer, timestep);
    }
    
    //lateral - inhibitory to excitatory
    std::cout << "\n\n.......\nSetting up inhibitory synapses...\n.......\n\n";

    //finish inhibitory parameter structure
    inh_synapse_params_vec->delay_range[0] = 10.0*timestep;
    inh_synapse_params_vec->delay_range[1] = 10.0*timestep; //NB that as the delays will be set later, these values are arbitrary, albeit required by Spike
    inh_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;

    //add plasticity rule
    CustomSTDPPlasticity * inhibitory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) inh_stdp_params);  
    PolyNetwork->AddPlasticityRule(inhibitory_stdp);
    
    for (int i=0; i<n_ex_layers; i++){
      pre_layer_ID = inh_layer_IDs[i];
      post_layer_ID = ex_layer_IDs[i];
      construct_connectivity(PolyNetwork, inh_to_ex, pre_layer_ID, post_layer_ID, inh_synapse_params_vec, inhibitory_stdp, inh_weight_min, inh_weight_max, n_neurons_per_layer, timestep);
    }
  }
 

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
  
  //create vector of rates for input layer
  std::vector<float> input_rates(x_dim*y_dim, input_firing_rate);

  //append background firing rates
  int background_input_size = x_dim*y_dim;
  for (int i = 0; i < background_input_size; i++){
    input_rates.push_back(background_firing_rate);
  }

  //print input stimulus
  std::cout << "Input:\n\n";
  for (int i = 0; i < x_dim; i++){
    for (int j = 0; j < y_dim; j++){
      std::cout << std::to_string(input_rates[i*x_dim+j]) << " ";
    }
    std::cout << "\n";
  }

  //print background stimulus
  std::cout << "\n\nBackground Input:\n\n";
  for (int i = 0; i < x_dim; i++){
    for (int j = 0; j < y_dim; j++){
      std::cout << std::to_string(input_rates[x_dim*y_dim+i*x_dim+j]) << " ";
    }
    std::cout << "\n";
  }

  //assign rates to stimulus
  int stimulus = patterned_poisson_input_neurons->add_stimulus(input_rates);
  std::cout << "\n\nNew stimulus ID is " << std::to_string(stimulus) << "\n";

  PolyNetwork->finalise_model();

  std::cout << "\n\n.......\nModel finalised and ready for simulating...\n.......\n\n";


  ////////////////////////
  // RUN WITH TRAINING ///
  ////////////////////////

  PolyNetwork->spiking_synapses->save_weights_as_txt(output_dir, "Initial_");

  // Loop through a certain number of epoch's of presentation
  for (int ii = 0; ii < training_epochs; ++ii) {
    patterned_poisson_input_neurons->select_stimulus(stimulus);
    PolyNetwork->run(simtime, 1); //the second argument is a boolean determining if STDP is on or off
    PolyNetwork->spiking_synapses->save_weights_as_txt(training_data_dir, "Epoch_" + std::to_string(ii) + "_");
    spike_monitor_main->save_spikes_as_binary(training_data_dir, "Epoch_" + std::to_string(ii) + "_Output_");
    spike_monitor_input->save_spikes_as_binary(training_data_dir, "Epoch_" + std::to_string(ii) + "_Input_");
    spike_monitor_main->reset_state(); //Dumps all recorded spikes
    spike_monitor_input->reset_state(); //Dumps all recorded spikes
    PolyNetwork->reset_time(); //Resets the internal clock to 0
  }

  

  ///////////////////////////
  // RUN WITHOUT TRAINING ///
  ///////////////////////////

  // Loop through a certain number of epochs of presentation
  for (int ii = 0; ii < display_epochs; ++ii) {

    patterned_poisson_input_neurons->select_stimulus(stimulus);
    PolyNetwork->run(simtime, 0); //the second argument is a boolean determining if STDP is on or off
    spike_monitor_main->save_spikes_as_binary(testing_data_dir, "Epoch_" + std::to_string(ii) +  "_Output_");
    spike_monitor_input->save_spikes_as_binary(testing_data_dir, "Epoch_" + std::to_string(ii) +  "_Input_");
    spike_monitor_main->reset_state(); //Dumps all recorded spikes
    spike_monitor_input->reset_state(); //Dumps all recorded spikes
    PolyNetwork->reset_time(); //Resets the internal clock to 0

  }




 std::cout << "\n\n.......\nFinished Simulation\n.......\n\n";



 
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//FUNCTION TO CONSTRUCT CONNECTIVITY

void construct_connectivity(SpikingModel* model, connectivity_struct connectivity, int pre_layer_ID, int post_layer_ID, conductance_spiking_synapse_parameters_struct* synapse_params, CustomSTDPPlasticity * stdp, float weight_min, float weight_max, int n_neurons_per_layer, float timestep)
{

  //prepare random number generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  int maximum_value_rand = generator.max();

  //prepare weight distribution
  std::uniform_real_distribution<double> weight_distribution(weight_min, weight_max);

  //choose connections
  for (int j = 0; j < n_neurons_per_layer; j++){ //cycle through pre-synaptic neuron IDs (NB: for the purpose of adding synapses, neurons indexed seperately for each layer, from 0)
    for (int k = 0; k < n_neurons_per_layer; k++){ //cycle through post-synaptic neuron IDs
      int r=generator(); //pick a random number between 0 and generator.max
      if (r < maximum_value_rand*connectivity.prob_connection){
        if (pre_layer_ID != post_layer_ID || j != k){ //make sure not to connect a neuron to itself
          for (int l = 0; l < connectivity.mult_synapses; l++){ //iterate through each synapse for this neuron pair
            synapse_params->pairwise_connect_presynaptic.push_back(j);
            synapse_params->pairwise_connect_postsynaptic.push_back(k);
          }
        }
      }
    }
  }

  //calculate number of synapses
  int n_synapses = synapse_params->pairwise_connect_presynaptic.size();

  //randomise delays
  float lognorm_mu = log(connectivity.delay_mean) - 0.5*log(connectivity.delay_std/pow(connectivity.delay_mean, 2) +1);
  float lognorm_std = sqrt(log(connectivity.delay_std)/pow(connectivity.delay_mean, 2) +1);
  std::lognormal_distribution<double> delay_distribution(lognorm_mu, lognorm_std);
  for (int j = 0; j <n_synapses; j++){
    synapse_params->pairwise_connect_delay.push_back(delay_distribution(generator)/1000 + timestep); //distribution is in seconds, so must be scaled to ms. a small value added to prevent floating point errors in spike
  }

  //randomise weights
  for (int j = 0; j <n_synapses; j++){
    synapse_params->pairwise_connect_weight.push_back(weight_distribution(generator));
  }

  //print sanity check
  std::cout << "Layer ID " << std::to_string(pre_layer_ID) << " is sending " << std::to_string(n_synapses) << " synapses to layer ID " << std::to_string(post_layer_ID) << "\n";

  //turn on learning if it ought to be
  if (connectivity.learning){
    synapse_params->plasticity_vec.push_back(stdp);
    std::cout << "Learning on\n";
  } else {
    std::cout << "Learning off\n";
  }


  //add synapse group
  model->AddSynapseGroup(pre_layer_ID, post_layer_ID, synapse_params);

  //clear vectors
  synapse_params->pairwise_connect_presynaptic.clear();
  synapse_params->pairwise_connect_postsynaptic.clear();
  synapse_params->pairwise_connect_weight.clear();
  synapse_params->pairwise_connect_delay.clear();
  synapse_params->plasticity_vec.clear();

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

