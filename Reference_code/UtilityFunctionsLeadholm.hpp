#ifndef UTILITY_FUNCS_LEAD
#define UTILITY_FUNCS_LEAD

/*
Utility functions for loading custom weights into the Spike simulator
*/

#include "Spike/Spike.hpp"
#include <array>
#include <iostream>
#include <cstring>

//Function for loading weights from binary, Python-generated connectivity file
//Note that connections, weights, and synaptic delays must all be specified in the input file
void connect_from_python(
  int layer1,
  int layer2,
  conductance_spiking_synapse_parameters_struct* SYN_PARAMS,
  std::string filename,
  SpikingModel* Model){

  ifstream connectivity_data;
  std::vector<float> input_IDs_vec_float;
  std::vector<float> output_IDs_vec_float;
  std::vector<float> synapse_weights_vec;
  std::vector<float> synapse_delays_vec;
  connectivity_data.open(filename.c_str(), ios::binary);

  if (connectivity_data.is_open()){
    std::cout << "Connectivity data file opened, extracting synapse IDs, weights and delays...\n";

    // Find the number of synapses
    connectivity_data.seekg(0, std::ios::end);
    int num_synapses = (connectivity_data.tellg() / sizeof(float))/4; //tellg will return the total number of bytes,
    // indicated by the final read position in the file, which was obtained by seekg and ::end in the line above, and corrected for the number
    // of data arrays (pre-IDs, post-IDs, weights, and delays)


    // Define the size of the synapse-data vectors
    input_IDs_vec_float.resize(num_synapses); // Resize the previously initialized vector, now that the number of elements is known
    output_IDs_vec_float.resize(num_synapses); // Resize the previously initialized vector, now that the number of elements is known
    synapse_weights_vec.resize(num_synapses); // Resize the previously initialized vector, now that the number of elements is known
    synapse_delays_vec.resize(num_synapses); // Resize the previously initialized vector, now that the number of elements is known

    // Extract pre-synaptic IDs
    connectivity_data.seekg(0, std::ios::beg); // Move stream to the beginning of the input
    connectivity_data.read(reinterpret_cast<char*>(&input_IDs_vec_float[0]), num_synapses*sizeof(float)); //char types are a single byte in C++, so the char pointer is a way of...
    // enforcing C++ to read the file byte-by-byte; this does not however change the data type of e.g. input_rates from whatever it was initialized to before
    std::vector<int> input_IDs_vec_int(input_IDs_vec_float.begin(), input_IDs_vec_float.end()); //Cast vector as int

    // Extract post-synaptic IDs
    connectivity_data.read(reinterpret_cast<char*>(&output_IDs_vec_float[0]), num_synapses*sizeof(float));
    std::vector<int> output_IDs_vec_int(output_IDs_vec_float.begin(), output_IDs_vec_float.end()); //Cast vector as int

    // Extract synaptic weights
    connectivity_data.read(reinterpret_cast<char*>(&synapse_weights_vec[0]), num_synapses*sizeof(float)); 

    // Extract synaptic delays
    connectivity_data.read(reinterpret_cast<char*>(&synapse_delays_vec[0]), num_synapses*sizeof(float));

    // Assign extracted data to the synapse object
    SYN_PARAMS->pairwise_connect_presynaptic = input_IDs_vec_int;
    SYN_PARAMS->pairwise_connect_postsynaptic = output_IDs_vec_int;
    SYN_PARAMS->pairwise_connect_weight = synapse_weights_vec;
    SYN_PARAMS->pairwise_connect_delay = synapse_delays_vec;

    // Add the newly defined synapses to the model object
    Model->AddSynapseGroup(layer1, layer2, SYN_PARAMS);
    std::cout << "Connectivity data file read.\n";
    
  }
  
  else{
  std::cout << "\nError: Issue opening file \n";
  }

  connectivity_data.close();

}

#endif // UTILITY_FUNCS_LEAD