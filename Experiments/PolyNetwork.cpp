#include "Spike/Spike.hpp"
#include "UtilityFunctionsLeadholm.hpp"
#include <array>
#include <iostream>
#include <cstring>
#include <string>


// The following is a network with a 'binary' architecture, in that it consists of just two streams of multi-layered, side-by-side processing
// The degree of interactivity between these two streams is determined by the user

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

  //Simulation parameters; these can be relatively easily adjusted to observe the affect on the network
  int training_epochs = 10; // Number of epochs to have STDP active
  int display_epochs = 10; // Number of epochs where the each stimulus is presented with STDP inactive
  float exc_inh_weight_ratio = 5.0; //parameter that determines how much stronger inhibitory synapses are than excitatory synapses
  int background_firing_rate = 15000; //Poisson firing rate (Hz) of noisy neurons feeding into all layers, and preventing dead neurons
  
  // Initialize core model parameters
  int x_dim = 5;
  int y_dim = 5;
  int input_firing_rate = 30; //approximate firing rate of input stimuli; note multiplier used later to generate actual stimuli
  //float competitive_connection_prob = 0.2; // Probability parameter that controls how the two competing halves of the network are connected
  float timestep = 0.0001;  // In seconds
  float lower_weight_limit = 0.005;
  float upper_weight_limit = 0.015; //For initiliazing weights
  float max_weight_parameter = 0.03; //Maximum value that can be learned
  PolyNetwork->SetTimestep(timestep);
   int n_layers = 3; //number of layers (not including input layer)

  // Choose an input neuron type
  PatternedPoissonInputSpikingNeurons* patterned_poisson_input_neurons = new PatternedPoissonInputSpikingNeurons();
  // Choose the neuron type
  LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
  // Choose the synapse type
  ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();

  // Allocate your chosen components to the simulator
  PolyNetwork->input_spiking_neurons = patterned_poisson_input_neurons;
  PolyNetwork->spiking_neurons = lif_spiking_neurons;
  PolyNetwork->spiking_synapses = conductance_spiking_synapses;

  // *** Allocate chosen plasticity rule
  custom_stdp_plasticity_parameters_struct * Excit_STDP_PARAMS = new custom_stdp_plasticity_parameters_struct;
  Excit_STDP_PARAMS->a_plus = 1.0f; //Set to the mean of the excitatory weight distribution
  Excit_STDP_PARAMS->a_minus = 1.0f;
  Excit_STDP_PARAMS->weight_dependence_power_ltd = 0.0f; //By setting this to 0, the STDP rule has *no* LTD weight dependence, and hence behaves like the classical Gerstner rule
  Excit_STDP_PARAMS->w_max = max_weight_parameter; //Sets the maximum weight that can be *learned* (hard border)
  Excit_STDP_PARAMS->tau_plus = 0.01f;
  Excit_STDP_PARAMS->tau_minus = 0.01f;
  Excit_STDP_PARAMS->learning_rate = 0.001f;
  Excit_STDP_PARAMS->a_star = 0; //Excit_STDP_PARAMS->a_plus * Excit_STDP_PARAMS->tau_minus * Inhib_STDP_PARAMS->targetrate;

  CustomSTDPPlasticity * excitatory_stdp = new CustomSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) patterned_poisson_input_neurons, (stdp_plasticity_parameters_struct *) Excit_STDP_PARAMS);  
  
  PolyNetwork->AddPlasticityRule(excitatory_stdp);

    /*
      ADD ANY ACTIVITY MONITORS
  */
  SpikingActivityMonitor* spike_monitor_main = new SpikingActivityMonitor(lif_spiking_neurons);
  PolyNetwork->AddActivityMonitor(spike_monitor_main);

  SpikingActivityMonitor* spike_monitor_input = new SpikingActivityMonitor(patterned_poisson_input_neurons);
  PolyNetwork->AddActivityMonitor(spike_monitor_input);


  // SETTING UP INPUT NEURONS

  // Creating an input neuron parameter structure
  patterned_poisson_input_spiking_neuron_parameters_struct* input_neuron_params = new patterned_poisson_input_spiking_neuron_parameters_struct();
    
  // Setting the dimensions of the input neuron layer
  input_neuron_params->group_shape[0] = 5;    // x-dimension of the input neuron layer
  input_neuron_params->group_shape[1] = 5;    // y-dimension of the input neuron layer

  //Create a group of input neurons. This function returns the ID of the input neuron group
  int input_layer_ID = PolyNetwork->AddInputNeuronGroup(input_neuron_params);
  std::cout << "New input layer ID is " << input_layer_ID << "\n";


  // SETTING UP BACKGROUND INPUT NEURONS

  // Creating a background input neuron parameter structure
  patterned_poisson_input_spiking_neuron_parameters_struct* back_input_neuron_params = new patterned_poisson_input_spiking_neuron_parameters_struct();

  // Setting the dimensions of the background input neuron layers
  back_input_neuron_params->group_shape[0] = x_dim;    // x-dimension of the input neuron layer
  back_input_neuron_params->group_shape[1] = y_dim;   // y-dimension of the input neuron layer

  // Iteratively create all the layers of background neurons
  vector<int> background_layer_IDs(n_layers, 0);
    for (int ii = 0; ii < n_layers; ii++){
        background_layer_IDs[ii]=PolyNetwork->AddInputNeuronGroup(back_input_neuron_params);
        std::cout << "New background layer ID is " << background_layer_IDs[ii] << "\n";
      }

  int total_number_of_input_neurons = x_dim*y_dim*(n_layers+1);
    
    
  // SETTING UP NEURON GROUPS
  // Creating an LIF parameter structure for an excitatory neuron population and an inhibitory

  lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
  excitatory_population_params->group_shape[0] = 5;
  excitatory_population_params->group_shape[1] = 5;
  excitatory_population_params->resting_potential_v0 = -0.06f;
  excitatory_population_params->absolute_refractory_period = 0.002f;
  excitatory_population_params->threshold_for_action_potential_spike = -0.05f;
  excitatory_population_params->somatic_capacitance_Cm = 200.0*pow(10, -12);
  excitatory_population_params->somatic_leakage_conductance_g0 = 10.0*pow(10, -9);
    
  /*lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
  inhibitory_population_params->group_shape[0] = 5;
  inhibitory_population_params->group_shape[1] = 5;
  inhibitory_population_params->resting_potential_v0 = -0.082f;
  inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
  inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
  inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);*/
    
  // Create populations of excitatory and inhibitory neurons
  vector<int> ex_layer_IDs(n_layers, 0);
  //vector<int> inh_layer_IDs(n_layers, 0);

  for (int i=0; i<(n_layers+1); i++){
    ex_layer_IDs[i] = PolyNetwork->AddNeuronGroup(excitatory_population_params);
    std::cout << "New ex layer ID is " << ex_layer_IDs[i] << "\n";
    //inh_layer_IDs[i] = PolyNetwork->AddNeuronGroup(inhibitory_population_params);
    //std::cout << "New inh layer ID is " << inh_layer_IDs[i] << "\n";
  }
    

  // SETTING UP SYNAPSES

  std::cout << "\n\n.......\nBuilding feed-forward connectivity...\n.......\n\n";

  // FEED-FORWARD CONNECTIONS

  int mult_synapses = 4;
  int weight_mean = 0.01/(10.0*pow(10, -9));
  int delay_mean = 3.4;

  conductance_spiking_synapse_parameters_struct * ff_synapse_params_vec = new conductance_spiking_synapse_parameters_struct();
  ff_synapse_params_vec->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
  ff_synapse_params_vec->delay_range[0] = 10.0*timestep;
  ff_synapse_params_vec->delay_range[1] = 10.0*timestep; //NB that as the delays will be set later from loaded files, these values are arbitrary, albeit required by Spike
  ff_synapse_params_vec->decay_term_tau_g = 0.0017f;  // Seconds (Conductance Parameter)
  ff_synapse_params_vec->reversal_potential_Vhat = 0.0*pow(10.0, -3);
  ff_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;
  //ff_synapse_params_vec->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
  //ff_synapse_params_vec->weight_range[1] = 10;
  ff_synapse_params_vec->plasticity_vec.push_back(excitatory_stdp);


  int n_neurons_per_layer = x_dim*y_dim;

  for(int j = 0; j < n_neurons_per_layer; j++){
    for (int k = 0; k < n_neurons_per_layer; k++){
      for (int l = 0; l < mult_synapses; l++){
        ff_synapse_params_vec->pairwise_connect_presynaptic.push_back(j);
        ff_synapse_params_vec->pairwise_connect_postsynaptic.push_back(k);
        ff_synapse_params_vec->pairwise_connect_weight.push_back(weight_mean);
        ff_synapse_params_vec->pairwise_connect_delay.push_back(delay_mean*timestep);
      }
    }
  }


  PolyNetwork->AddSynapseGroup(input_layer_ID, 0, ff_synapse_params_vec);
  ff_synapse_params_vec->pairwise_connect_presynaptic.clear();
  ff_synapse_params_vec->pairwise_connect_postsynaptic.clear();
  ff_synapse_params_vec->pairwise_connect_weight.clear();
  ff_synapse_params_vec->pairwise_connect_delay.clear();

  for (int i = 0; i < n_layers; i++){
    for (int j = 0; j < n_neurons_per_layer; j++){
      for (int k = 0; k < n_neurons_per_layer; k++){
        for (int l = 0; l < mult_synapses; l++){
          ff_synapse_params_vec->pairwise_connect_presynaptic.push_back(j);
          ff_synapse_params_vec->pairwise_connect_postsynaptic.push_back(k);
          ff_synapse_params_vec->pairwise_connect_weight.push_back(weight_mean);
          ff_synapse_params_vec->pairwise_connect_delay.push_back(delay_mean*timestep);
        }
      }
    }
    PolyNetwork->AddSynapseGroup(i, i+1, ff_synapse_params_vec);
    ff_synapse_params_vec->pairwise_connect_presynaptic.clear();
    ff_synapse_params_vec->pairwise_connect_postsynaptic.clear();
    ff_synapse_params_vec->pairwise_connect_weight.clear();
    ff_synapse_params_vec->pairwise_connect_delay.clear();
  }


  //PolyNetwork->AddSynapseGroup(input_layer_ID, 0, ff_synapse_params_vec);

  //for (int i = 0; i < n_layers; i++){
  //  PolyNetwork->AddSynapseGroup(i, i+1, ff_synapse_params_vec);
  //}


  
  // BACKGROUND CONNECTIONS

  std::cout << "\n\n.......\nBuilding background connectivity...\n.......\n\n";

  //Create synapses for the background noise input to all other neurons
  conductance_spiking_synapse_parameters_struct * back_input_to_all = new conductance_spiking_synapse_parameters_struct();
  back_input_to_all->weight_range[0] = lower_weight_limit;   // Create uniform distributions of weights between the upper and lower bound
  back_input_to_all->weight_range[1] = upper_weight_limit; //NB the weight range is simply the initialization
  back_input_to_all->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
  back_input_to_all->delay_range[0] = 10.0*timestep;
  back_input_to_all->delay_range[1] = 10.0*timestep;
  back_input_to_all->decay_term_tau_g = 0.005f;  // Seconds (Conductance Parameter)
  back_input_to_all->reversal_potential_Vhat = 0.0*pow(10.0, -3);
  back_input_to_all->connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;

  // Iteratively create the background input synapses
  for (int ii = 0; ii < n_layers; ii++){ // Iterate through the layers
    //Note the first (input) layers are skipped, and each side of each layer only needs to receive one connection
    PolyNetwork->AddSynapseGroup(background_layer_IDs[ii], ex_layer_IDs[ii+1], back_input_to_all);
    std::cout << "\n\nBackground input neurons " << background_layer_IDs[ii] << " are sending to group ID " << ex_layer_IDs[ii+1] << "\n";
  
  }

  std::cout << "\n\n.......\nAll synapses created...\n.......\n\n";

  /*
      ADD INPUT STIMULI TO THE PATTERNED POISSON NEURONS CLASS
  */

  std::cout << "\n\n.......\nAssigning stimuli to the network...\n.......\n\n";

  int total_input_plus_background_size = (x_dim * y_dim * (n_layers+1));
  
  std::vector<float> input_rates(total_number_of_input_neurons, input_firing_rate);

  for (int i = total_input_plus_background_size+1; i < total_number_of_input_neurons; i++){
    input_rates.push_back(background_firing_rate);
  }

  int stimulus = patterned_poisson_input_neurons->add_stimulus(input_rates);


  PolyNetwork->finalise_model();
  float simtime = 0.2f; //This should be long enough to allow any recursive signalling to finish propagating

  std::cout << "\n\n.......\nModel finalized and ready for simulating...\n.......\n\n";

  
  //    RUN THE SIMULATION WITH TRAINING
  
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

  

  /*
      TEST
  */

  // Loop through a certain number of epoch's of presentation
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