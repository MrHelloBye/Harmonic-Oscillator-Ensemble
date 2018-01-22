import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("output.tsv", delimiter='\t', skiprows=1)

time = np.transpose(data[:,0])

part_num = int((data.shape[1]-1)/2)
pos_tracks = []
mom_tracks = []
for i in range(1,part_num+1):
    pos_tracks.append(np.transpose(data[:,i]))
    mom_tracks.append(np.transpose(data[:,i+part_num]))


for track in pos_tracks:
    plt.plot(time,track)
plt.xlabel("Time")
plt.ylabel("Position")
plt.savefig("Positions.pdf")
plt.show()

for track in mom_tracks:
    plt.plot(time,track)
plt.xlabel("Time")
plt.ylabel("Momentum")
plt.savefig("Momenta.pdf")
plt.show()



for i in range(len(pos_tracks)):
    plt.plot(pos_tracks[i],mom_tracks[i])
    #-i*10/(part_num+1)
plt.xlabel("Positions")
plt.ylabel("Momenta")
plt.savefig("Phase Portrait.pdf")
plt.show()


avg_pos = np.zeros_like(pos_tracks[0])
avg_mom = np.zeros_like(pos_tracks[0])
for i in range(len(pos_tracks)):
    avg_pos += pos_tracks[i]
    avg_mom += mom_tracks[i]
avg_pos /= part_num
avg_mom /= part_num
plt.plot(avg_pos,avg_mom)
plt.xlabel("Positions")
plt.ylabel("Momenta")
plt.tight_layout()
plt.savefig("Total Phase Portrait.pdf")
plt.show()