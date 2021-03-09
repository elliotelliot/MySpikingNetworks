
#include <iostream>
#include <string>
#include <random>

int pick_pre_synaptic_ID(int post_synaptic_ID, int fan_in, int n_pre_synaptic_neurons, int n_post_synaptic_neurons)
{
    
    int expansion_factor = ceil( (float) n_pre_synaptic_neurons/ (float)n_post_synaptic_neurons);
    float adjustment_factor;
    if (n_post_synaptic_neurons > n_pre_synaptic_neurons){
        adjustment_factor = (float)n_pre_synaptic_neurons / (float)n_post_synaptic_neurons;
    }
    else{
        adjustment_factor = 1.0;
    }
    
    int start_ID = expansion_factor * post_synaptic_ID - fan_in;
    int end_ID = expansion_factor * post_synaptic_ID + expansion_factor - 1 + fan_in;
    
    if (start_ID < 0){
        start_ID = 0;
    }
    
    if (end_ID > (n_pre_synaptic_neurons / adjustment_factor - 1)){
        end_ID = n_pre_synaptic_neurons / adjustment_factor - 1;
    }
    
    for (int i = start_ID; i < (end_ID+1); i++){
        std::cout << floor(i * adjustment_factor) << " ";
    }
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(start_ID, end_ID);
    int neuron_ID = floor(distribution(generator) * adjustment_factor);
    return(neuron_ID);
 

}
        

int main()
{
  
  int n_pre_synaptic_neurons = 2;
  int n_post_synaptic_neurons = 4;
  int fan_in = 0;
  int neuron_ID;
  
  int expansion_factor = ceil( (float) n_pre_synaptic_neurons/ (float)n_post_synaptic_neurons);
  std::cout << "expansion factor: " << expansion_factor;
  
  for (int i = 0; i<(n_post_synaptic_neurons); i++){
      std::cout << "\npost-synaptic ID: " << i << "\npre-synaptic IDs: ";
      neuron_ID = pick_pre_synaptic_ID(i, fan_in, n_pre_synaptic_neurons, n_post_synaptic_neurons);
      std::cout << "\nselected neuron: " << neuron_ID;
  }
  
  return 0;
  
}
