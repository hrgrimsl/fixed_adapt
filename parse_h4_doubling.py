from sys import argv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
script, fname = argv
params = 1
Energies = []
E0s = []
Done = False
#df = pd.read_csv('h4_300/adapt.csv')
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
params = 1
Done = False
while Done == False:
   e0s = []
   os.system(f"grep -A16 \"^Parameters: {params}$\" {fname} > temp.dat")
   if os.path.getsize("temp.dat") == 0 or os.path.exists("temp.dat") == False:
       Done = True
   else:
       os.system(f"grep \"^Initial Energy:\" temp.dat > temp2.dat")
       with open("temp2.dat", "r") as f:
           for i in f.readlines():
               e0s.append(float(i.split()[-1]))        
       E0s.append(e0s)
       params += 1

Seeds = []
Done = False
params = 1
while Done == False:
   seeds = []
   os.system(f"grep -A16 \"^Parameters: {params}$\" {fname} > temp.dat")
   if os.path.getsize("temp.dat") == 0 or os.path.exists("temp.dat") == False:
       Done = True
   else:
       os.system(f"grep \"^Initialization:\" temp.dat > temp2.dat")
       with open("temp2.dat", "r") as f:
           for i in f.readlines():
               try:
                   seeds.append(int(i.split()[-1]))
               except:
                   seeds.append("Recycled") 
       Seeds.append(seeds)
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
#plt.plot(xs, np.array(df['Energy'])-ci, color = "red", label = "ADAPT-Style Optimization w/ Same Operators")
#plt.plot(xs, np.array(df['SED Energy'])-ci, color = "black", label = "Subspace ED")
plt.scatter(xs, [float("NaN") for i in xs], color = "black", label = "Random Initializations")
plt.vlines(19, 0, 1, color = "black", label = "Last new operator before ansatz is repeated.")
plt.legend()

plt.xlabel("ADAPT Iterations")
plt.ylabel("Error From ED (a.u.)")
plt.yscale("symlog", linthresh = 1e-10)
plt.title("ADAPT's Local Minima")
plt.show()
''' 
x0s = []
for seed in Seeds[0]:
    if seed != "Recycled":
        np.random.seed(seed)
        x0s.append(math.pi*np.sqrt(2)*np.random.rand(1)[0])
    else:
        x0s.append(0.0)
plt.scatter(x0s , E0s[0])
plt.show()
x0s = []
y0s = []
for seed in Seeds[1]:
    if seed != "Recycled":
        np.random.seed(seed)
        x0s.append(math.pi*np.sqrt(2)*np.random.rand(2)[0])
        y0s.append(math.pi*np.sqrt(2)*np.random.rand(2)[1])
    else:
        x0s.append(0)
        y0s.append(0)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
idx = np.argsort(E0s[1])
ax = fig.add_subplot(111,projection = '3d')
ax.plot_surface(np.array(x0s),np.array(y0s), zs = np.array(E0s[1])[idx])
plt.show()

plt.violinplot(list(np.array(Energies)-ci), positions = [i+1 for i in range(0, params-1)])
plt.plot(xs, np.array(best_es)-ci, color = "blue", label = "Best Initialization")
plt.plot(xs, np.array(recycled_es)-ci, color = "green", label = "Recycled Initialization")
plt.legend()
plt.xlabel("ADAPT Iterations")
plt.ylabel("Error From ED (a.u.)")
plt.yscale("log")
plt.title("ADAPT's Local Minima")
plt.show() 
'''
