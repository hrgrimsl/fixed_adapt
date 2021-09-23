from sys import argv
import os
import matplotlib.pyplot as plt
import numpy as np

script, fname = argv
params = 1
Energies = []
Done = False

os.system(f"grep -A1 \"^Reference information:$\" {fname} > temp.dat")
with open("temp.dat", "r") as f:
   for line in f.readlines():
       if line.split()[0] == "Reference":
           pass
       else:
           hf = float(line.split()[-1])
os.system(f"grep -A1 \"^ED Solution 1:$\" {fname} > temp.dat")
with open("temp.dat", "r") as f:
   for line in f.readlines():
       if line.split()[0] == "ED":
           pass
       else:
           ci = float(line.split()[-1])
print(hf)
print(ci)

while Done == False:
   energies = []
   os.system(f"grep -A16 \"^Parameters: {params}$\" {fname} > temp.dat")
   if os.path.getsize("temp.dat") == 0 or os.path.exists("temp.dat") == False:
       Done = True
   else:
       os.system(f"grep \"^Final Energy:\" temp.dat > temp2.dat")
       with open("temp2.dat", "r") as f:
           for i in f.readlines():
               energies.append(float(i.split()[-1]))        
       Energies.append(energies)
       params += 1

xs = [0] 
best_es = [hf]
recycled_es = [hf]
for i in range(0, params-1): 
    plt.scatter((i+1)*np.ones(len(Energies[i])), np.array(Energies[i])-ci, c = [i for i in range(0, len(Energies[i]))], alpha = .3, cmap = 'Spectral_r')
    best_es.append(np.amin(np.array(Energies[i])))
    xs.append(i+1)
os.system(f"grep -A15 \"^Initialization: Recycled$\" {fname} > temp.dat")
with open("temp.dat", "r") as f:
    for line in f.readlines():
        if line.split()[0] == "Final":
            recycled_es.append(float(line.split()[-1]))

plt.plot(xs, np.array(best_es)-ci, color = "blue", label = "Best Initialization")
plt.plot(xs, np.array(recycled_es)-ci, color = "green", label = "Recycled Initialization")
plt.scatter(xs, [float("NaN") for i in xs], color = "black", label = "Random Initializations")

plt.legend()
plt.xlabel("ADAPT Iterations")
plt.ylabel("Error From ED (a.u.)")
plt.yscale("log")
plt.title("ADAPT's Local Minima")
plt.show() 

