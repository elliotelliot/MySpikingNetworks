
/*
 
 An Example Model for running the SPIKE simulator
 
 To create the executable for this network:
 - Run cmake from the build directory: "cmake ../"
 - Make this example: "make ExampleExperiment"
 - Finally, execute the binary: "./ExampleExperiment"
 
 
 */


#include "Spike/Spike.hpp"

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
    
    /*
     CHOOSE THE COMPONENTS OF YOUR SIMULATION
     */
    
    // Create an instance of the Model
    SpikingModel* SimpleNetwork = new SpikingModel();
    
    
    // Set up the simulator with a timestep at which the neuron, synapse and STDP properties will be calculated
    float timestep = 0.0001;  // In seconds
    SimpleNetwork->SetTimestep(timestep);
    
    
    // Choose an input neuron type
    //GeneratorInputSpikingNeurons* generator_input_neurons = new GeneratorInputSpikingNeurons();
    PoissonInputSpikingNeurons * poisson_input_neurons = new PoissonInputSpikingNeurons();
    
    // Choose your neuron type
    LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
    
    // Choose your synapse type
    ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
    // VoltageSpikingSynapses * voltage_spiking_synapses = new VoltageSpikingSynapses();
    // CurrentSpikingSynapses * current_spiking_synapses = new CurrentSpikingSynapses();
    
    // Allocate your chosen components to the simulator
    SimpleNetwork->input_spiking_neurons = poisson_input_neurons;
    SimpleNetwork->spiking_neurons = lif_spiking_neurons;
    SimpleNetwork->spiking_synapses = conductance_spiking_synapses;
    
    /*
     ADD ANY ACTIVITY MONITORS OR PLASTICITY RULES YOU WISH FOR
     */
    SpikingActivityMonitor* spike_monitor = new SpikingActivityMonitor(lif_spiking_neurons);
    SpikingActivityMonitor* input_spike_monitor = new SpikingActivityMonitor(poisson_input_neurons);
    SimpleNetwork->AddActivityMonitor(spike_monitor);
    SimpleNetwork->AddActivityMonitor(input_spike_monitor);
    
    /*
     SETUP PROPERTIES AND CREATE NETWORK:
     
     Note:
     All Neuron, Synapse and STDP types have associated parameters structures.
     These structures are defined in the header file for that class and allow us to set properties.
     */
    
    
    // SETTING UP NEURON GROUPS
    // Creating an LIF parameter structure for an excitatory neuron population and an inhibitory
    lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
    excitatory_population_params->group_shape[0] = 5;
    excitatory_population_params->group_shape[1] = 5;
    excitatory_population_params->resting_potential_v0 = -0.074f;
    excitatory_population_params->threshold_for_action_potential_spike = -0.053f;
    excitatory_population_params->somatic_capacitance_Cm = 500.0*pow(10, -12);
    excitatory_population_params->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);
    
    lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
    inhibitory_population_params->group_shape[0] = 5;
    inhibitory_population_params->group_shape[1] = 5;
    inhibitory_population_params->resting_potential_v0 = -0.082f;
    inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
    inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
    inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
    
    // Create populations of excitatory and inhibitory neurons
    int n_layers = 4; //number of layers (not including input layer)
    vector<int> ex_layer_IDs(n_layers, 0);
    vector<int> inh_layer_IDs(n_layers, 0);

    for (int i=0; i<(n_layers); i++){
        ex_layer_IDs[i] = SimpleNetwork->AddNeuronGroup(excitatory_population_params);
        std::cout << "New ex layer ID is " << ex_layer_IDs[i] << "\n";
        inh_layer_IDs[i] = SimpleNetwork->AddNeuronGroup(inhibitory_population_params);
        std::cout << "New inh layer ID is " << inh_layer_IDs[i] << "\n";
    }

    // SETTING UP INPUT NEURONS
    // Creating an input neuron parameter structure
    poisson_input_spiking_neuron_parameters_struct* input_neuron_params = new poisson_input_spiking_neuron_parameters_struct();
    // Setting the dimensions of the input neuron layer
    input_neuron_params->group_shape[0] = excitatory_population_params->group_shape[0];;    // x-dimension of the input neuron layer
    input_neuron_params->group_shape[1] = excitatory_population_params->group_shape[1];    // y-dimension of the input neuron layer
    input_neuron_params->rate = 50.0f;

    //Create a group of input neurons. This function returns the ID of the input neuron group
    int input_layer_ID = SimpleNetwork->AddInputNeuronGroup(input_neuron_params);
    std::cout << "New background input layer ID is " << input_layer_ID << "\n";

    
    /*// Create a group of input neurons for every layer. This function returns the ID of the input neuron group
    vector<int> input_layer_IDs(n_layers, 0);
    for (int i=0; i<(n_layers+1); i++){
        input_layer_IDs[i] = SimpleNetwork->AddInputNeuronGroup(input_neuron_params);
        std::cout << "New background input layer ID is " << input_layer_IDs[i] << "\n";
    }*/
    
    
    // SETTING UP SYNAPSES
    // Creating a synapses parameter structure for connections from the input neurons to the excitatory neurons
    conductance_spiking_synapse_parameters_struct* input_to_excitatory_parameters = new conductance_spiking_synapse_parameters_struct();
    input_to_excitatory_parameters->weight_range[0] = 10.0f;   // Create uniform distributions of weights [0.5, 10.0]
    input_to_excitatory_parameters->weight_range[1] = 10.0f;
    input_to_excitatory_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
    input_to_excitatory_parameters->delay_range[0] = timestep;    // Create uniform distributions of delays [1 timestep, 0.1 secs]
    input_to_excitatory_parameters->delay_range[1] = 5*timestep;
    //input_to_excitatory_parameters->random_connectivity_probability = 0.5;
    input_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;

    //Feedforward excitatory connections
    conductance_spiking_synapse_parameters_struct* ff_synapse_parameters = new conductance_spiking_synapse_parameters_struct();
    ff_synapse_parameters->weight_range[0] = 0.5f;   // Create uniform distributions of weights [0.5, 10.0]
    ff_synapse_parameters->weight_range[1] = 10.0f;
    ff_synapse_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
    ff_synapse_parameters->delay_range[0] = timestep;    // Create uniform distributions of delays [1 timestep, 10ms]
    ff_synapse_parameters->delay_range[1] = 0.01f;
    ff_synapse_parameters->random_connectivity_probability = 0.1f;
    ff_synapse_parameters->connectivity_type = CONNECTIVITY_TYPE_RANDOM;


    //Creating a set of synapse parameters for connections from the excitatory neurons to the inhibitory neurons
    conductance_spiking_synapse_parameters_struct * excitatory_to_inhibitory_parameters = new conductance_spiking_synapse_parameters_struct();
    excitatory_to_inhibitory_parameters->weight_range[0] = 10.0f;
    excitatory_to_inhibitory_parameters->weight_range[1] = 10.0f;
    excitatory_to_inhibitory_parameters->weight_scaling_constant = inhibitory_population_params->somatic_leakage_conductance_g0;
    excitatory_to_inhibitory_parameters->delay_range[0] = 5*timestep;
    excitatory_to_inhibitory_parameters->delay_range[1] = 3.0f*pow(10, -3);
    excitatory_to_inhibitory_parameters->random_connectivity_probability = 1;
    excitatory_to_inhibitory_parameters->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
    
    // Creating a set of synapse parameters from the inhibitory neurons to the excitatory neurons
    conductance_spiking_synapse_parameters_struct * inhibitory_to_excitatory_parameters = new conductance_spiking_synapse_parameters_struct();
    inhibitory_to_excitatory_parameters->weight_range[0] = -5.0f;
    inhibitory_to_excitatory_parameters->weight_range[1] = -2.5f;
    inhibitory_to_excitatory_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
    inhibitory_to_excitatory_parameters->delay_range[0] = 0.01f;
    inhibitory_to_excitatory_parameters->delay_range[1] = 0.05f;
    inhibitory_to_excitatory_parameters->random_connectivity_probability = 1;
    inhibitory_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
    
    
    // CREATING SYNAPSES
    // When creating synapses, the ids of the presynaptic and postsynaptic populations are all that are required
    // Note: Input neuron populations cannot be post-synaptic on any synapse


    SimpleNetwork->AddSynapseGroup(input_layer_ID, ex_layer_IDs[0], input_to_excitatory_parameters); //Input to first excitatory layer


    for (int i = 0; i<(n_layers-1); i++){
        SimpleNetwork->AddSynapseGroup(ex_layer_IDs[i], ex_layer_IDs[i+1], ff_synapse_parameters);
    }

    for (int i = 0; i<n_layers; i++){
        SimpleNetwork->AddSynapseGroup(ex_layer_IDs[i], inh_layer_IDs[i], excitatory_to_inhibitory_parameters);
        SimpleNetwork->AddSynapseGroup(ex_layer_IDs[i], inh_layer_IDs[i], inhibitory_to_excitatory_parameters);
        //SimpleNetwork->AddSynapseGroup(input_layer_IDs[i], ex_layer_IDs[i], input_to_excitatory_parameters); //Input to every excitatory layer
    }
    
    /*
     ADD INPUT STIMULI TO THE GENERATOR NEURONS CLASS
     */
    
    // We can now assign a set of spike times to neurons in the input layer
    /*int s1_num_spikes = 5;
    int s1_neuron_ids[5] = {0, 1, 3, 6, 7};
    float s1_spike_times[5] = {0.0001f, 0.0003f, 0.0002f, 0.0005f, 0.0009f};
    // Adding this stimulus to the input neurons
    int first_stimulus = generator_input_neurons->add_stimulus(s1_num_spikes, s1_neuron_ids, s1_spike_times);
    // Creating a second stimulus
    int s2_num_spikes = 5;
    int s2_neuron_ids[5] = {2, 5, 9, 8, 0};
    float s2_spike_times[5] = {0.0001f, 0.0003f, 0.0002f, 0.0005f, 0.0009f};
    int second_stimulus = generator_input_neurons->add_stimulus(s2_num_spikes, s2_neuron_ids, s2_spike_times);*/

    
    
    /*
     RUN THE SIMULATION
     */
    
    // The only argument to run is the number of seconds
    SimpleNetwork->finalise_model();
    float simtime = 1.0f;
    //generator_input_neurons->select_stimulus(first_stimulus);
    SimpleNetwork->run(simtime);
    
   //generator_input_neurons->select_stimulus(second_stimulus);
    //SimpleNetwork->run(simtime);
    
    
    spike_monitor->save_spikes_as_txt("./");
    input_spike_monitor->save_spikes_as_txt("./", "Input");
    //SimpleNetwork->spiking_synapses->save_connectivity_as_txt("./");
    
    return 0;
}

