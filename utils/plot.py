import matplotlib.pyplot as plt
import pandas as pd

name = "../plots/0.1r_noOutScale_bone_gating_256h_fc_record"
data = pd.read_csv(name+".txt", header=None, sep=' ', dtype='float')

plt.plot(data.index, data[1], 'y*-')
plt.plot(data.index, data[2], 'r--')
plt.savefig(name+"_plot.png")
plt.show()
