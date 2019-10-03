# MySpikingNetworks

Repository with Spike \(from github.com/nasiryahm/Spike\) included as a submodule, for easy creation of new spiking networks without editing main Spike code, and use of Niels\'s method for calling connectivity data \(see github.com/nielsleadholm/PolyNetwork_CppTools for basis\)


**To clone:**

>git clone --recursive https://github.com/elliotelliot/MySpikingNetworks


**To compile:**

>cmake ./
>make -j8


**To run:**

*Cleanup:*

>rm \*.bin
>rm \*SpikeTimes.txt
>rm \*SpikeIDs.txt

*Run:*

>./BinaryNetwork
