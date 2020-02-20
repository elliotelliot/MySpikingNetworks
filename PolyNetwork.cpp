#include "Spike/Spike.hpp"
#include "UtilityFunctionsLeadholm.hpp"
#include <array>
#include <iostream>
#include <cstring>
#include <string>
#include <random>
#include <chrono>


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

  // Create an instance of the Model
  SpikingModel* PolyNetwork = new SpikingModel();
  /* Explanation of above notation:
    PolyNetwork is intiliazed as a pointer to an object of class SpikingModel
    The 'new' operator is essentially the C++ equivalent of 'malloc' allocates memory for the un-named object, and returns the pointer to this object,
    or if it is an array, the first element. The memory allocation performed by new is with 'dynamic storage duration', such that the lifetime of the 
    object isn't limited to the scope in which it was created. This is also known as allocating memory to the 'heap' (as opposed to the stack)
    and as such memory *de*-allocation is critical in order to prevent a memory leak/'garbage' building up
  */

  
  /////////////////
  // PARAMETERS ///
  /////////////////
  
  //SIMULATION
  int training_epochs = 10; // Number of epochs to have STDP active
  int display_epochs = 10; // Number of epochs where the each stimulus is presented with STDP inactive
  float timestep = 0.0001;  // In seconds
  int input_firing_rate = 50; //Poisson firing rate (Hz) of input neurons feeding into first layer
  int background_firing_rate = 500; // Poisson firing rate (Hz) of noisy neurons feeding into all layers, preventing dead neurons
  float exc_inh_weight_ratio = 1.0; //parameter that determines how much stronger inhibitory synapses are than excitatory synapses

  //NETWORK ARCHITECTURE
  int x_dim = 5;
  int y_dim = 5;
  int n_layers = 3; //number of layers (not including input layer)

  //CONNECTIVITY
  /*Niels used:
	--Background input = ata, 1 synapse per connection, delay range 10*timestep:100*timestep, learning off
	--Inhibitory lateral = random connectivity, 1 synapse per connection, delay = 1ms, weights scaled *5, learning off
	--Excitatory lateral = 
	--Feed-forward = ata, 4 synapses per connection, learning on
	--Main input = same as feed-forward
	*/
  int mult_synapses = 4; //number of synapses per connection
  float delay_mean = 3.4f; float delay_std = 2.3f; //mean/std axonal delay - follows lognorm dist
  float weight_min = 0.005; float weight_max = 0.015; //initial weights min/max - follows uniform dist
  int prob_ff_connection = 1; //probability of a feedforward connection between any neuron pair in adjacent layers
  int prob_lat_connection = 1; //probability of a connection between any neuron pair within a layer
  int prob_inh_to_ex_connection = 1; //probability of a connection from any given inhibitory neuron to any given excitatory neuron within a layer
  int prob_ex_to_inh_connection = 1; //probability of a connection from any given excitatory neuron to any given inhibitory neuron within a layer
  int prob_input_connection = 1; //probability of a connection from any given input neuron to any given excitatory neuron within the first layer
  int prob_background_connection = 1; //probability of a connection from any given background input neuron to any given excitatory neuron within a layer

  //INPUT NEURONS
  PatternedPoissonInputSpikingNeurons* patterned_poisson_input_neurons = new PatternedPoissonInputSpikingNeurons();
  PolyNetwork->input_spiking_neurons = patterned_poisson_input_neurons;

  //MAIN NEURONS
  LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
  PolyNetwork->spiking_neurons = lif_spiking_neurons;

  lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
  excitatory_population_params->group_shape[0] = x_dim;
  excitatory_population_params->group_shape[1] = y_dim;
  excitatory_population_params->resting_potential_v0 = -0.06f;
  excitatory_population_params->absolute_refractory_period = 0.002f;
  excitatory_population_params->threshold_for_action_potential_spike = -0.05f;
  excitatory_population_params->somatic_capacitance_Cm = 200.0*pow(10, -12);
  excitatory_population_params->somatic_leakage_conductance_g0 = 10.0*pow(10, -9);
    
  lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
  inhibitory_population_params->group_shape[0] = x_dim;
  inhibitory_population_params->group_shape[1] = y_dim;
  inhibitory_population_params->resting_potential_v0 = -0.082f;
  inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
  inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
  inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
  
  //SYNAPSES
  ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
  PolyNetwork->spiking_synapses = conductance_spiking_synapses;

  conductance_spiking_synapse_parameters_struct * synapse_params_vec = new conductance_spiking_synapse_parameters_struct();
  synapse_params_vec->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
  synapse_params_vec->delay_range[0] = 10.0*timestep;
  synapse_params_vec->delay_range[1] = 10.0*timestep; //NB that as the delays will be set later, these values are arbitrary, albeit required by Spike
  synapse_params_vec->decay_term_tau_g = 0.0017f;  // Seconds (Conductance Parameter)
  synapse_params_vec->reversal_potential_Vhat = 0.0*pow(10.0, -3);
  synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;

  //PLASTICITY
  custom_stdp_plasticity_parameters_struct * Excit_STDP_PARAMS = new custom_stdp_plasticity_parameters_struct;
  Excit_STDP_PARAMS->a_plus = 1.0f; //Set to the mean of the excitatory weight distribution
  Excit_STDP_PARAMS->a_minus = 1.0f;
  Excit_STDP_PARAMS->weight_dependence_power_ltd = 0.0f; //By setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  Excit_STDP_PARAMS->w_max = 0.03; //Sets the maximum weight that can be *learned* (hard border)
  Excit_STDP_PARAMS->tau_plus = 0.01f;
  Excit_STDP_PARAMS->tau_minus = 0.01f;
  Excit_STDP_PARAMS->learning_rate = 0.001f;
  Excit_STDP_PARAMS->a_star = 0; //Excit_STDP_PARAMS->a_plus * Excit_STDP_PARAMS->tau_minus * Inhib_STDP_PARAMS->targetrate;

  CustomSTDPPlasticity * excitatory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) Excit_STDP_PARAMS);  

  //ACTIVITY MONITORS
  SpikingActivityMonitor* spike_monitor_main = new SpikingActivityMonitor(lif_spiking_neurons);
  SpikingActivityMonitor* spike_monitor_input = new SpikingActivityMonitor(patterned_poisson_input_neurons);

  //ALLOCATE TO MODEL
  PolyNetwork->SetTimestep(timestep);
  PolyNetwork->AddActivityMonitor(spike_monitor_main);
  PolyNetwork->AddActivityMonitor(spike_monitor_input);
  PolyNetwork->AddPlasticityRule(excitatory_stdp);

  

  ///////////////////////////
  // SET UP NEURON GROUPS ///
  ///////////////////////////

  std::cout << "\n\n.......\nSetting up neuron groups...\n.......\n\n";


  // SETTING UP INPUT NEURONS

  //create new parameter structure
  patterned_poisson_input_spiking_neuron_parameters_struct* input_neuron_params = new patterned_poisson_input_spiking_neuron_parameters_struct();
  input_neuron_params->group_shape[0] = x_dim;    // x-dimension of the input neuron layer
  input_neuron_params->group_shape[1] = y_dim;    // y-dimension of the input neuron layer

  //create a group of input neurons. This function returns the ID of the input neuron group
  int input_layer_ID = PolyNetwork->AddInputNeuronGroup(input_neuron_params);
  std::cout << "New input layer ID is " << input_layer_ID << "\n";

  
  // SETTING UP BACKGROUND INPUT NEURONS

  //create new parameter structure
  patterned_poisson_input_spiking_neuron_parameters_struct* back_input_neuron_params = new patterned_poisson_input_spiking_neuron_parameters_struct();
  back_input_neuron_params->group_shape[0] = x_dim;    // x-dimension of the input neuron layer
  back_input_neuron_params->group_shape[1] = y_dim;   // y-dimension of the input neuron layer

  //iteratively create all the layers of background neurons (one for each main layer)
  vector<int> background_layer_IDs(n_layers, 0);
    for (int ii = 0; ii < n_layers; ii++){
        background_layer_IDs[ii]=PolyNetwork->AddInputNeuronGroup(back_input_neuron_params);
        std::cout << "New background layer ID is " << background_layer_IDs[ii] << "\n";
      }
    
    
  // SETTING UP MAIN GROUPS
  
  //initialise vectors to store layer IDs
  vector<int> ex_layer_IDs(n_layers, 0);
  vector<int> inh_layer_IDs(n_layers, 0);

  //iteratively add each layer to network
  for (int i=0; i<n_layers; i++){
    ex_layer_IDs[i] = PolyNetwork->AddNeuronGroup(excitatory_population_params);
    std::cout << "New excitatory layer ID is " << ex_layer_IDs[i] << "\n";
    inh_layer_IDs[i] = PolyNetwork->AddNeuronGroup(inhibitory_population_params);
    std::cout << "New inhibitory layer ID is " << inh_layer_IDs[i] << "\n";
  }

 
  //////////////////////
  // SET UP SYNAPSES ///
  //////////////////////

  std::cout << "\n\n.......\nSetting up synapses...\n.......\n\n";

  //DETERMINE CONNECTIVITY

  //initialise vectors
  std::vector<int> pre_layer_IDs;
  std::vector<int> post_layer_IDs;
  std::vector<float> prob_connection;
  std::vector<int> learning;
  std::vector<int> scaling;
  
  //input to first layer (ex)
  pre_layer_IDs.push_back(input_layer_ID);
  post_layer_IDs.push_back(ex_layer_IDs[0]);
  prob_connection.push_back(prob_input_connection);
  learning.push_back(0);
  scaling.push_back(1);

  //feed-forward (ex)
  for (int i=0; i<(n_layers-1); i++){
    pre_layer_IDs.push_back(ex_layer_IDs[i]);
    post_layer_IDs.push_back(ex_layer_IDs[(i+1)]);
    prob_connection.push_back(prob_ff_connection);
    learning.push_back(1);
    scaling.push_back(1);
  }

  //lateral (within ex layer)
  for (int i=0; i<n_layers; i++){
    pre_layer_IDs.push_back(ex_layer_IDs[i]);
    post_layer_IDs.push_back(ex_layer_IDs[i]);
    prob_connection.push_back(prob_lat_connection);
    learning.push_back(1);
    scaling.push_back(1);
  }

  //lateral (ex->inh)
  for (int i=0; i<n_layers; i++){
    pre_layer_IDs.push_back(ex_layer_IDs[i]);
    post_layer_IDs.push_back(inh_layer_IDs[i]);
    prob_connection.push_back(prob_inh_to_ex_connection);
    learning.push_back(1);
    scaling.push_back(1);
  }

  //lateral (inh->ex)
  for (int i=0; i<n_layers; i++){
    pre_layer_IDs.push_back(inh_layer_IDs[i]);
    post_layer_IDs.push_back(ex_layer_IDs[i]);
    prob_connection.push_back(prob_ex_to_inh_connection);
    learning.push_back(1);
    scaling.push_back(exc_inh_weight_ratio);
  }

  //background (ex)
  for (int i=0; i<n_layers; i++){
    pre_layer_IDs.push_back(background_layer_IDs[i]);
    post_layer_IDs.push_back(ex_layer_IDs[i]);
    prob_connection.push_back(prob_background_connection);
    learning.push_back(0);
    scaling.push_back(1);
  }

  //PREPARE

  //prepare random number generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  //prepare delay distribution
  float lognorm_mu = log(delay_mean) - 0.5*log(delay_std/pow(delay_mean, 2) +1);
  float lognorm_std = sqrt(log(delay_std)/pow(delay_mean, 2) +1);
  std::lognormal_distribution<double> delay_distribution(lognorm_mu, lognorm_std);

  //prepare weight distribution
  std::uniform_real_distribution<double> weight_distribution(weight_min, weight_max);

  //calculate relevant sizes
  int n_neurons_per_layer = x_dim*y_dim;
  int n_synapse_groups = pre_layer_IDs.size();

  //SET UP SYNAPSES

  for (int i = 0; i < n_synapse_groups; i++){ //cycle through synapse groups

    //turn learning on if it should be
    if (learning[i] == 1){
      synapse_params_vec->plasticity_vec.push_back(excitatory_stdp);
    }
    
    //choose connections
    for (int j = 0; j < n_neurons_per_layer; j++){ //cycle through pre-synaptic neuron IDs (NB: for the purpose of adding synapses, neurons indexed seperately for each layer, from 0)
      	for (int k = 0; k < n_neurons_per_layer; k++){ //cycle through post-synaptic neuron IDs
        int r = rand(); //pick a random number between 0 and rd.max
        	if (r < RAND_MAX*prob_connection[i]){
          		if (pre_layer_IDs[i] != post_layer_IDs[i] || j != k){ //make sure not to connect a neuron to itself
          			for (int l = 0; l < mult_synapses; l++){ //iterate through each synapse for this neuron pair
						synapse_params_vec->pairwise_connect_presynaptic.push_back(j);
            			synapse_params_vec->pairwise_connect_postsynaptic.push_back(k);
          			}
          		}
        	}
      	}
    }

    //calculate number of synapses
    int n_synapses = synapse_params_vec->pairwise_connect_presynaptic.size();

    //randomise delays
    for (int j = 0; j <n_synapses; j++){
      synapse_params_vec->pairwise_connect_delay.push_back(delay_distribution(generator)/1000 + timestep); //distribution is in seconds, so must be scaled to ms. a small value added to prevent floating point errors in spike
    }

    //randomise weights
    for (int j = 0; j <n_synapses; j++){
      synapse_params_vec->pairwise_connect_weight.push_back(scaling[i]*weight_distribution(generator));
    }

    //print sanity check
    std::cout << "Layer ID " << std::to_string(pre_layer_IDs[i]) << " is sending " << std::to_string(n_synapses) << " synapses to layer ID " << std::to_string(post_layer_IDs[i]) << "\n";

    //add synapse group
    PolyNetwork->AddSynapseGroup(pre_layer_IDs[i], post_layer_IDs[i], synapse_params_vec);

    //clear vectors
    synapse_params_vec->pairwise_connect_presynaptic.clear();
    synapse_params_vec->pairwise_connect_postsynaptic.clear();
    synapse_params_vec->pairwise_connect_weight.clear();
    synapse_params_vec->pairwise_connect_delay.clear();
    synapse_params_vec->plasticity_vec.clear();
  }


  /////////////////////
  // SET UP STIMULI ///
  /////////////////////

  std::cout << "\n\n.......\nAssigning stimuli to the network...\n.......\n\n";
  
  //create vector of rates for input layer
  std::vector<float> input_rates(x_dim*y_dim, input_firing_rate);

  //append background firing rates
  int background_input_size = x_dim*y_dim*n_layers;
  for (int i = 0; i < background_input_size; i++){
    input_rates.push_back(background_firing_rate);
  }

  //assign rates to stimulus
  int stimulus = patterned_poisson_input_neurons->add_stimulus(input_rates);
  std::cout << "New stimulus ID is " << std::to_string(stimulus) << "\n";

  PolyNetwork->finalise_model();
  float simtime = 0.2f; //This should be long enough to allow any recursive signalling to finish propagating

  std::cout << "\n\n.......\nModel finalized and ready for simulating...\n.......\n\n";

  

  ////////////////////////
  // RUN WITH TRAINING ///
  ////////////////////////
  
  PolyNetwork->spiking_synapses->save_weights_as_txt("./Outputs/", "Initial_");

  // Loop through a certain number of epoch's of presentation
  for (int ii = 0; ii < training_epochs; ++ii) {
    patterned_poisson_input_neurons->select_stimulus(stimulus);
    PolyNetwork->run(simtime, 1); //the second argument is a boolean determining if STDP is on or off
    PolyNetwork->spiking_synapses->save_weights_as_txt("./Outputs/", "Epoch_" + std::to_string(ii) + "_");
    spike_monitor_main->save_spikes_as_txt("./Outputs/", "Epoch_" + std::to_string(ii) + "_Output_Training_");
    spike_monitor_input->save_spikes_as_txt("./Outputs/", "Epoch_" + std::to_string(ii) + "_Input_Training_");
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
    spike_monitor_main->save_spikes_as_txt("./Outputs/", "Epoch_" + std::to_string(ii) +  "_Output_Testing_");
    spike_monitor_input->save_spikes_as_txt("./Outputs/", "Epoch_" + std::to_string(ii) +  "_Input_Testing_");
    spike_monitor_main->reset_state(); //Dumps all recorded spikes
    spike_monitor_input->reset_state(); //Dumps all recorded spikes
    PolyNetwork->reset_time(); //Resets the internal clock to 0

  }


 
  return 0;
}