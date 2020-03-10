#include "Spike/Spike.hpp"
#include "UtilityFunctionsLeadholm.hpp"
#include <array>
#include <iostream>
#include <cstring>
#include <string>
#include <random>
#include <chrono>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*

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


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

  // Create an instance of the Model
  SpikingModel* PolyNetwork = new SpikingModel();
  
  /////////////////
  // PARAMETERS ///
  /////////////////
  
  //SIMULATION
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
  float ex_resting_potential = -0.06f; //v0
  float ex_absolute_refractory_period = 0.002f;
  float ex_threshold = -0.05f;
  float ex_somatic_capacitance = 200.0*pow(10, -12); //Cm
  float ex_somatic_leakage_conductance = 18.0*pow(10, -9); //g0
  
  //INHIBITORY NEURON DYNAMICS
  float inh_resting_potential = -0.082f; //v0
  float inh_threshold = -0.053f;
  float inh_somatic_capacitance = 214.0*pow(10, -12); //Cm
  float inh_somatic_leakage_conductance = 18.0*pow(10, -9); //g0

  //CONNECTIVITY
  //Input/feedforward
  int ff_mult_synapses = 4; //number of synapses per excitatory connection
  float ff_delay_mean = 3.4f; float ff_delay_std = 2.3f; //mean/std axonal delay - follows lognorm dist
  int prob_input_connection = 1; //probability of a connection from any given input neuron to any given excitatory neuron within the first layer
  int prob_background_connection = 1; //probability of a connection from any given background neuron to any given excitatory neuron within each layer
  int prob_ff_connection = 1; //probability of a feedforward connection between any neuron pair in adjacent layers
  //Feedback
  int fb_mult_synapses = 4; //number of synapses per excitatory connection
  float fb_delay_mean = 3.4f; float fb_delay_std = 2.3f; //mean/std axonal delay - follows lognorm dist
  int prob_fb_connection = 1; //probability of a feedback connection between any neuron pair in adjacent layers
  //Lateral excitatory
  int lat_mult_synapses = 4; //number of synapses per excitatory connection
  float lat_delay_mean = 3.4f; float lat_delay_std = 2.3f; //mean/std axonal delay - follows lognorm dist
  int prob_lat_connection = 1; //probability of a connection between any neuron pair within a layer
  int prob_ex_to_inh_connection = 1; //probability of a connection from any excitatory neuron to any inhibitory neuron in corresponding layer
  //Lateral inhibitory
  int inh_mult_synapses = 4; //number of synapses per excitatory connection
  float inh_delay_mean = 3.4f; float inh_delay_std = 2.3f; //mean/std axonal delay - follows lognorm dist
  int prob_inh_to_ex_connection = 1;
  
  //EXCITATORY SYNAPTIC DYNAMICS
  float ex_weight_min = 0.005; float ex_weight_max = 0.015; //initial weights min/max - follows uniform dist
  conductance_spiking_synapse_parameters_struct * ex_synapse_params_vec = new conductance_spiking_synapse_parameters_struct();  //create excitatory synapses parameter structure
  ex_synapse_params_vec->decay_term_tau_g = 0.0017f;  //conductance parameter (seconds)
  ex_synapse_params_vec->reversal_potential_Vhat = 0.0*pow(10.0, -3); //v_hat
  ex_synapse_params_vec->weight_scaling_constant = ex_somatic_leakage_conductance;

  //INHIBITORY SYNAPTIC DYNAMICS
  float inh_weight_min = 0.005; float inh_weight_max = 0.015; //initial weights min/max - follows uniform dist
  conductance_spiking_synapse_parameters_struct * inh_synapse_params_vec = new conductance_spiking_synapse_parameters_struct();
  inh_synapse_params_vec->decay_term_tau_g = 0.0017f; //conductance parameter (seconds)
  inh_synapse_params_vec->reversal_potential_Vhat = -80.0*pow(10.0, -3); //v_hat
  inh_synapse_params_vec->weight_scaling_constant = ex_somatic_leakage_conductance*5.0; //inhibitory weights scaled to be greater than excitatory weights

  //EXCITATORY PLASTICITY
  bool input_learning = 1; //is input -> first layer stdp on? 1=yes, 0=no
  bool background_learning = 0; //is background -> all main layers stdp on? 1=yes, 0=no
  bool ff_learning = 1; //is feed-forward stdp on? 1=yes, 0=no
  bool fb_learning = 1; //is feedback stdp on? 1=yes, 0=no
  bool lat_learning = 1; //is lateral excitatory stdp on? 1=yes, 0=no
  bool ex_to_inh_learning = 1; //is excitatory -> inhibitory stdp on? 1=yes, 0=no
  custom_stdp_plasticity_parameters_struct * ex_stdp_params = new custom_stdp_plasticity_parameters_struct(); //create excitatory stdp parameter structure
  ex_stdp_params->a_plus = 1.0f; //set to the mean of the excitatory weight distribution
  ex_stdp_params->a_minus = 1.0f;
  ex_stdp_params->weight_dependence_power_ltd = 0.0f; //by setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  ex_stdp_params->w_max = 0.03; //sets the maximum weight that can be *learned* (hard border)
  ex_stdp_params->tau_plus = 0.01f;
  ex_stdp_params->tau_minus = 0.01f;
  ex_stdp_params->learning_rate = 0.001f;
  ex_stdp_params->a_star = 0; //ex_stdp_params->a_plus * ex_stdp_params->tau_minus * Inhib_STDP_PARAMS->targetrate;

  //INHIBITORY PLASTICITY
  bool inh_learning = 1; //is lateral inhibitory -> excitatory stdp on? 1=yes, 0=no
  custom_stdp_plasticity_parameters_struct * inh_stdp_params = new custom_stdp_plasticity_parameters_struct(); //create inhibitory stdp parameter structure
  inh_stdp_params->a_plus = 1.0f; //set to the mean of the inhibitory weight distribution
  inh_stdp_params->a_minus = 1.0f;
  inh_stdp_params->weight_dependence_power_ltd = 0.0f; //by setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  inh_stdp_params->w_max = 0.03; //sets the maximum weight that can be *learned* (hard border)
  inh_stdp_params->tau_plus = 0.01f;
  inh_stdp_params->tau_minus = 0.01f;
  inh_stdp_params->learning_rate = 0.001f;
  inh_stdp_params->a_star = 0; //ex_stdp_params->a_plus * ex_stdp_params->tau_minus * Inhib_STDP_PARAMS->targetrate;
  

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////
  // SET UP NEURON GROUPS ///
  ///////////////////////////

  std::cout << "\n\n.......\nSetting up neuron groups...\n.......\n\n";

  //set the timestep first
  PolyNetwork->SetTimestep(timestep);

  // SETTING UP INPUT NEURONS

  //choose input neuron type
  PatternedPoissonInputSpikingNeurons* patterned_poisson_input_neurons = new PatternedPoissonInputSpikingNeurons();
  PolyNetwork->input_spiking_neurons = patterned_poisson_input_neurons;

  //create input parameter structure
  patterned_poisson_input_spiking_neuron_parameters_struct* input_neuron_params = new patterned_poisson_input_spiking_neuron_parameters_struct();
  input_neuron_params->group_shape[0] = x_dim;    // x-dimension of the input neuron layer
  input_neuron_params->group_shape[1] = y_dim;    // y-dimension of the input neuron layer

  //create a group of input neurons. This function returns the ID of the input neuron group
  int input_layer_ID = PolyNetwork->AddInputNeuronGroup(input_neuron_params);
  std::cout << "New input layer ID is " << input_layer_ID << "\n";

  //create a group of background input neurons
  int background_layer_ID = PolyNetwork->AddInputNeuronGroup(input_neuron_params);
  std::cout << "New background input layer ID is " << background_layer_ID << "\n";

  // SETTING UP MAIN GROUPS

  //choose neuron type
  LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
  PolyNetwork->spiking_neurons = lif_spiking_neurons;
  
  //initialise vector to store layer IDs
  vector<int> ex_layer_IDs(n_ex_layers, 0);
    
  //create excitatory parameter structure
  lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
  excitatory_population_params->group_shape[0] = x_dim;
  excitatory_population_params->group_shape[1] = y_dim;
  excitatory_population_params->resting_potential_v0 = ex_resting_potential;
  excitatory_population_params->absolute_refractory_period = ex_absolute_refractory_period;
  excitatory_population_params->threshold_for_action_potential_spike = ex_threshold;
  excitatory_population_params->somatic_capacitance_Cm = ex_somatic_capacitance;
  excitatory_population_params->somatic_leakage_conductance_g0 = ex_somatic_leakage_conductance;
    
  //iteratively add each excitatory layer to network
  for (int i=0; i<n_ex_layers; i++){
    ex_layer_IDs[i] = PolyNetwork->AddNeuronGroup(excitatory_population_params);
    std::cout << "New excitatory layer ID is " << ex_layer_IDs[i] << "\n";
  }

  //initialise vector to store layer IDs
  vector<int> inh_layer_IDs(n_ex_layers, 0);

  //if there are inhbitory layers
  if (inhibition == 1){

    //create inhibitory parameter structure
    lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
    inhibitory_population_params->group_shape[0] = x_dim;
    inhibitory_population_params->group_shape[1] = y_dim;
    inhibitory_population_params->resting_potential_v0 = inh_resting_potential;
    inhibitory_population_params->threshold_for_action_potential_spike = inh_threshold;
    inhibitory_population_params->somatic_capacitance_Cm = inh_somatic_capacitance;
    inhibitory_population_params->somatic_leakage_conductance_g0 = inh_somatic_leakage_conductance;

    //iteratively add each inhibitory layer to network
    for (int i=0; i<n_ex_layers; i++){
      inh_layer_IDs[i] = PolyNetwork->AddNeuronGroup(inhibitory_population_params);
      std::cout << "New inhibitory layer ID is " << inh_layer_IDs[i] << "\n";
    }
  }



 
  /////////////////////////////////
  // SET UP EXCITATORY SYNAPSES ///
  /////////////////////////////////

  std::cout << "\n\n.......\nSetting up excitatory synapses...\n.......\n\n";

  //choose synapse type (nb this will be the same for inhibitory synapses too)
  ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
  PolyNetwork->spiking_synapses = conductance_spiking_synapses;

  //finish excitatory parameter structure
  ex_synapse_params_vec->delay_range[0] = 10.0*timestep;
  ex_synapse_params_vec->delay_range[1] = 10.0*timestep; //NB that as the delays will be set later, these values are arbitrary, albeit required by Spike
  ex_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;

  //assign plasticity rule
  CustomSTDPPlasticity * excitatory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) ex_stdp_params);  
  PolyNetwork->AddPlasticityRule(excitatory_stdp);

  //initialise vectors
  std::vector<int> pre_layer_IDs;
  std::vector<int> post_layer_IDs;
  std::vector<float> prob_connection;
  std::vector<float> delay_means;
  std::vector<float> delay_stds;
  std::vector<int> mult_synapses;
  std::vector<bool> learning;
  
  //input to first layer
  pre_layer_IDs.push_back(input_layer_ID);
  post_layer_IDs.push_back(ex_layer_IDs[0]);
  prob_connection.push_back(prob_input_connection);
  delay_means.push_back(ff_delay_mean);
  delay_stds.push_back(ff_delay_std);
  mult_synapses.push_back(ff_mult_synapses);
  learning.push_back(input_learning);

  //background to all layers
  for (int i=0; i<n_ex_layers; i++){
    pre_layer_IDs.push_back(background_layer_ID);
    post_layer_IDs.push_back(ex_layer_IDs[i]);
    prob_connection.push_back(prob_background_connection);
    delay_means.push_back(ff_delay_mean);
    delay_stds.push_back(ff_delay_std);
    mult_synapses.push_back(ff_mult_synapses);
    learning.push_back(background_learning);
  }

  //feed-forward
  for (int i=0; i<(n_ex_layers-1); i++){
    pre_layer_IDs.push_back(ex_layer_IDs[i]);
    post_layer_IDs.push_back(ex_layer_IDs[(i+1)]);
    prob_connection.push_back(prob_ff_connection);
    delay_means.push_back(ff_delay_mean);
    delay_stds.push_back(ff_delay_std);
    mult_synapses.push_back(ff_mult_synapses);
    learning.push_back(ff_learning);
  }

  //feed-back
  for (int i=0; i<(n_ex_layers-1); i++){
    pre_layer_IDs.push_back(ex_layer_IDs[i+1]);
    post_layer_IDs.push_back(ex_layer_IDs[i]);
    prob_connection.push_back(prob_fb_connection);
    delay_means.push_back(fb_delay_mean);
    delay_stds.push_back(fb_delay_std);
    mult_synapses.push_back(fb_mult_synapses);
    learning.push_back(fb_learning);
  }

  //lateral - excitatory to excitatory
  for (int i=0; i<n_ex_layers; i++){
    pre_layer_IDs.push_back(ex_layer_IDs[i]);
    post_layer_IDs.push_back(ex_layer_IDs[i]);
    prob_connection.push_back(prob_lat_connection);
    delay_means.push_back(lat_delay_mean);
    delay_stds.push_back(lat_delay_std);
    mult_synapses.push_back(lat_mult_synapses);
    learning.push_back(lat_learning);
  }

  //lateral - excitatory to inhibitory
  if (inhibition == 1){
    for (int i=0; i<n_ex_layers; i++){
      pre_layer_IDs.push_back(ex_layer_IDs[i]);
      post_layer_IDs.push_back(inh_layer_IDs[i]);
      prob_connection.push_back(prob_ex_to_inh_connection);
      delay_means.push_back(lat_delay_mean);
      delay_stds.push_back(lat_delay_std);
      mult_synapses.push_back(lat_mult_synapses);
      learning.push_back(ex_to_inh_learning);
    }
  }
  
  //prepare random number generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  int maximum_value_rand = generator.max();

  //prepare weight distribution
  std::uniform_real_distribution<double> weight_distribution(ex_weight_min, ex_weight_max);

  //calculate relevant sizes
  int n_neurons_per_layer = x_dim*y_dim;
  int n_synapse_groups = pre_layer_IDs.size();

  //create synapses
  for (int i = 0; i < n_synapse_groups; i++){ //cycle through synapse groups

    //choose connections
    for (int j = 0; j < n_neurons_per_layer; j++){ //cycle through pre-synaptic neuron IDs (NB: for the purpose of adding synapses, neurons indexed seperately for each layer, from 0)
      for (int k = 0; k < n_neurons_per_layer; k++){ //cycle through post-synaptic neuron IDs
        int r=generator(); //pick a random number between 0 and generator.max
        if (r < maximum_value_rand*prob_connection[i]){
          if (pre_layer_IDs[i] != post_layer_IDs[i] || j != k){ //make sure not to connect a neuron to itself
          	for (int l = 0; l < mult_synapses[i]; l++){ //iterate through each synapse for this neuron pair
						  ex_synapse_params_vec->pairwise_connect_presynaptic.push_back(j);
            	ex_synapse_params_vec->pairwise_connect_postsynaptic.push_back(k);
          	}
          }
        }
      }
    }

    //calculate number of synapses
    int n_synapses = ex_synapse_params_vec->pairwise_connect_presynaptic.size();

    //randomise delays
    float lognorm_mu = log(delay_means[i]) - 0.5*log(delay_stds[i]/pow(delay_means[i], 2) +1);
    float lognorm_std = sqrt(log(delay_stds[i])/pow(delay_means[i], 2) +1);
    std::lognormal_distribution<double> delay_distribution(lognorm_mu, lognorm_std);
    for (int j = 0; j <n_synapses; j++){
      ex_synapse_params_vec->pairwise_connect_delay.push_back(delay_distribution(generator)/1000 + timestep); //distribution is in seconds, so must be scaled to ms. a small value added to prevent floating point errors in spike
    }

    //randomise weights
    for (int j = 0; j <n_synapses; j++){
      ex_synapse_params_vec->pairwise_connect_weight.push_back(weight_distribution(generator));
    }

    //print sanity check
    std::cout << "Layer ID " << std::to_string(pre_layer_IDs[i]) << " is sending " << std::to_string(n_synapses) << " synapses to layer ID " << std::to_string(post_layer_IDs[i]) << "\n";

    //turn on learning if it ought to be
    if (learning[i]){
      ex_synapse_params_vec->plasticity_vec.push_back(excitatory_stdp);
      std::cout << "Learning on\n";
    } else {
      std::cout << "Learning off\n";
    }


    //add synapse group
    PolyNetwork->AddSynapseGroup(pre_layer_IDs[i], post_layer_IDs[i], ex_synapse_params_vec);

    //clear vectors
    ex_synapse_params_vec->pairwise_connect_presynaptic.clear();
    ex_synapse_params_vec->pairwise_connect_postsynaptic.clear();
    ex_synapse_params_vec->pairwise_connect_weight.clear();
    ex_synapse_params_vec->pairwise_connect_delay.clear();
    ex_synapse_params_vec->plasticity_vec.clear();
  }
  

  /////////////////////////////////
  // SET UP INHIBITORY SYNAPSES ///
  /////////////////////////////////

  if (inhibition == 1){ //if there is inhibition

    std::cout << "\n\n.......\nSetting up inhibitory synapses...\n.......\n\n";

    //finish inhibitory parameter structure
    inh_synapse_params_vec->delay_range[0] = 10.0*timestep;
    inh_synapse_params_vec->delay_range[1] = 10.0*timestep; //NB that as the delays will be set later, these values are arbitrary, albeit required by Spike
    inh_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;

    //prepare delay distribution
    float lognorm_mu = log(inh_delay_mean) - 0.5*log(inh_delay_std/pow(inh_delay_mean, 2) +1);
    float lognorm_std = sqrt(log(inh_delay_std)/pow(inh_delay_mean, 2) +1);
    std::lognormal_distribution<double> delay_distribution(lognorm_mu, lognorm_std);

    //prepare weight distribution
    std::uniform_real_distribution<double> inh_weight_distribution(inh_weight_min, inh_weight_max);

    //assign plasticity rule
    CustomSTDPPlasticity * inhibitory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) inh_stdp_params);  
    PolyNetwork->AddPlasticityRule(inhibitory_stdp);

    for (int i = 0; i < n_ex_layers; i++){

      //choose connections
      for (int j = 0; j < n_neurons_per_layer; j++){ //cycle through pre-synaptic neuron IDs (NB: for the purpose of adding synapses, neurons indexed seperately for each layer, from 0)
        for (int k = 0; k < n_neurons_per_layer; k++){ //cycle through post-synaptic neuron IDs
          int r = rand(); //pick a random number between 0 and rd.max
          if (r < RAND_MAX*prob_connection[i]){
            for (int l = 0; l < inh_mult_synapses; l++){ //iterate through each synapse for this neuron pair
              inh_synapse_params_vec->pairwise_connect_presynaptic.push_back(j);
              inh_synapse_params_vec->pairwise_connect_postsynaptic.push_back(k);
            }
          }
        }
      }
    
      //calculate number of synapses
      int n_synapses = inh_synapse_params_vec->pairwise_connect_presynaptic.size();

      //randomise delays
      for (int j = 0; j <n_synapses; j++){
        inh_synapse_params_vec->pairwise_connect_delay.push_back(delay_distribution(generator)/1000 + timestep); //distribution is in seconds, so must be scaled to ms. a small value added to prevent floating point errors in spike
      }

      //randomise weights
      for (int j = 0; j <n_synapses; j++){
        inh_synapse_params_vec->pairwise_connect_weight.push_back(inh_weight_distribution(generator));
      }

      //print sanity check
      std::cout << "Layer ID " << std::to_string(inh_layer_IDs[i]) << " is sending " << std::to_string(n_synapses) << " synapses to layer ID " << std::to_string(ex_layer_IDs[i]) << "\n";

      //turn on learning if it ought to be
      if (inh_learning){
        inh_synapse_params_vec->plasticity_vec.push_back(inhibitory_stdp);
        std::cout << "Learning on\n";
      } else {
        std::cout << "Learning off\n";
      }

      //add synapse group
      PolyNetwork->AddSynapseGroup(inh_layer_IDs[i], ex_layer_IDs[i], inh_synapse_params_vec);

      //clear vectors
      inh_synapse_params_vec->pairwise_connect_presynaptic.clear();
      inh_synapse_params_vec->pairwise_connect_postsynaptic.clear();
      inh_synapse_params_vec->pairwise_connect_weight.clear();
      inh_synapse_params_vec->pairwise_connect_delay.clear();

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
  
  PolyNetwork->spiking_synapses->save_weights_as_txt("./Outputs/", "Initial_");

  // Loop through a certain number of epoch's of presentation
  for (int ii = 0; ii < training_epochs; ++ii) {
    patterned_poisson_input_neurons->select_stimulus(stimulus);
    PolyNetwork->run(simtime, 1); //the second argument is a boolean determining if STDP is on or off
    PolyNetwork->spiking_synapses->save_weights_as_txt("./Outputs/", "Epoch_" + std::to_string(ii) + "_");
    spike_monitor_main->save_spikes_as_binary("./Outputs/", "Epoch_" + std::to_string(ii) + "_Output_Training_");
    spike_monitor_input->save_spikes_as_binary("./Outputs/", "Epoch_" + std::to_string(ii) + "_Input_Training_");
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
    spike_monitor_main->save_spikes_as_binary("./Outputs/", "Epoch_" + std::to_string(ii) +  "_Output_Testing_");
    spike_monitor_input->save_spikes_as_binary("./Outputs/", "Epoch_" + std::to_string(ii) +  "_Input_Testing_");
    spike_monitor_main->reset_state(); //Dumps all recorded spikes
    spike_monitor_input->reset_state(); //Dumps all recorded spikes
    PolyNetwork->reset_time(); //Resets the internal clock to 0

  }


 
  return 0;
}