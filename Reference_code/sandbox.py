# def one():
#     return "January"
  
# def two():
#     return "February"

# def three():
#     return "March"

# def numbers_to_months(argument):
# 	switcher = {
# 		1: one,
# 		2: two,
# 		3: three
# 	}
# 	month = switcher.get(argument, lambda: "Invalid month")
# 	print(month)
  
# numbers_to_months(one)

import numpy as np


# ids = np.array([1, 2, 3, 4, 5, 6])
# times = np.array([0, 1, 5, 10, 3, 5.5])

# mintime = 3
# maxtime = 6

# high_pass_mask = times >= mintime
# print(high_pass_mask)
# low_pass_mask = times <= maxtime
# print(low_pass_mask)
# mask = np.where(high_pass_mask*low_pass_mask)
# print(mask)
# ids = ids[mask]
# print(ids)
# times = times[mask]
# print(times)

# total_items = 5000

# data = [1] * total_items

# items_per_group = 1000

# for i in range(total_items):

# 	a = data[(i/items_per_group)]


# myarray1 = np.zeros(5)
# myarray2 = np.ones(4)
# mylist1 = [myarray1, myarray2]
# mylist2 = [myarray2*3, myarray2*2]
# masterlist = [mylist1, mylist2]

# with open("file.py", "w") as output:
# 	output.write("masterlist = [")
# 	for i in range(len(masterlist)):
# 		mylist = masterlist[i]
# 		output.write("[")
# 		for j in range(len(mylist)):
# 			output.write("np.array(" + str(mylist[j]) + "), ")
# 		output.write("]")
# 	output.write("]")


# from file import *

# print(masterlist[0][1][1])

# array = masterlist[0][1]
# # array = array*2
# # print(array[1])

# simtime = 10.0
# n_time_bins = 5
# time_bin_width = simtime/n_time_bins

# spike_times = np.array([0, 1.5, 2, 3, 4, 5, 6, 8, 10])
# time_series = []
# for i in range(n_time_bins):
# 	spikes = spike_times[(spike_times >= time_bin_width*i) & (spike_times < time_bin_width*(i+1))]
# 	print(spikes)
# 	rate = float(len(spikes)/time_bin_width)
# 	time_series.append(rate)

# print(time_series)


# ids = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6])
# times = np.array([0, 1, 3, 4, 1, 3.4, 5, 9, 5, 10, 3, 5.5, 1.4])

# times_by_neuron = []
# for neuron_id in range(7): # Loop through each neuron

# 	times_for_this_neuron = times[np.where(ids == neuron_id)] # Create array of times when this neuron spiked in this epoch
# 	times_for_this_neuron = times_for_this_neuron.tolist()
# 	times_by_neuron.append(times_for_this_neuron)

# print(times_by_neuron)


import json

l = [1, 2, 3, 4]

with open("test.txt", "w") as fp:
	json.dump(l, fp)

with open("test.txt", "r") as fp:
	b = json.load(fp)

print(b)