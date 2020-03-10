import numpy as np
import pandas as pd #Pandas has multiple functions, including providing 'data_frame' objects that can be used for visualizing and analyzing data
from matplotlib import pyplot as plt
plt.switch_backend('agg')

df = pd.DataFrame(
  data = {
      "ids": np.fromfile("../Outputs/Epoch_0_Output_Testing_SpikeIDs.bin", dtype=np.int32),
      "times": np.fromfile("../Outputs/Epoch_0_Output_Testing_SpikeTimes.bin", dtype=np.float32),
  }
)

#NB Pandas will generate a 'data-frame', where each row in this case has a name (ids or times), and the columns in those rows contain the values of interest

plt.figure(figsize=(12,9))
#mask = stim1_df["times"] <1.0 #Restrict plotted spikes to a particular time period
#mask = ((stim1_df["ids"] > 0) & (stim1_df["ids"] <= 1024)) #Restrict plotted spikes to a particular layer
plt.scatter(df["times"], df["ids"], s=5)
plt.savefig('Figures/raster')
plt.close()
