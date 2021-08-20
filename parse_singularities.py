import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
from sys import argv
import re
import os

script, fname = argv

chem = 1/627.5094740631
scf = -1.97060224603216
ci = -2.8009588996544403

hessians = []
os.system(f"grep -A1 \"Hessian \" {fname} > temp.dat")

f = open("temp.dat", "r")
for line in f.readlines():
    if "Hessian" not in line and "--" not in line:
        hessians.append([float(i) for i in line.split(",")[:-1]])


max_length = len(hessians[-1])
sorted_hessians = []
for i in range(1, max_length+1):
    sorted_hessians.append([])
    for j in hessians:
        if len(j) == i:
            sorted_hessians[-1].append(abs(np.array(j)))

xprev = 0
print("Ops,VQE Steps")
for i in range(0, len(sorted_hessians)):
    print(f"{i},{len(sorted_hessians[i])},{abs(sorted_hessians[i][0][-1]/sorted_hessians[i][0][0])}")
    for j in range(0, len(sorted_hessians[i])):
        idx = np.argsort(sorted_hessians[i][j])
        idx = np.argsort(idx)  
        x = xprev+j*np.ones(len(sorted_hessians[i][j]))
        xprev += j

        #plt.scatter(x, sorted_hessians[i][j], c = idx, s = 3, cmap = "Spectral_r")        

       
xprev = 0
for i in range(0, max_length-1):    
    ith = sorted_hessians[i]
    colors = plt.cm.jet(np.linspace(0,1,i+1))
    for j in range(0, len(ith[0])):
        js = []
        x = [xprev + j for j in range(0, len(ith))]

        for h in ith:
            js.append(h[j])
        plt.plot(x, js, color = colors[j], linewidth = 1)
    xprev += len(ith)-1 
    plt.axvline(x = xprev, linewidth = .25, color = "black")


plt.axvline(x = 0, linewidth = .25, color = "black", label = "ADAPT Iteration Demarcations")
plt.title("SVD Spectrum of Hessian Throughout ADAPT-VQE")
plt.xlabel("Cumulative VQE Iterations")
plt.ylabel("Hessian Eigenvalues")
plt.yscale("log")

plt.show()


hessians = []
os.system(f"grep -A1 \"Jacobian \" {fname} > temp.dat")

f = open("temp.dat", "r")
for line in f.readlines():
    if "Hessian" not in line and "--" not in line:
        hessians.append([float(i) for i in line.split(",")[:-1]])


max_length = len(hessians[-1])
sorted_hessians = []
for i in range(1, max_length+1):
    sorted_hessians.append([])
    for j in hessians:
        if len(j) == i:
            sorted_hessians[-1].append(abs(np.array(j)))

xprev = 0
print("Ops,VQE Steps")
for i in range(0, len(sorted_hessians)):
    print(f"{i},{len(sorted_hessians[i])},{abs(sorted_hessians[i][0][-1]/sorted_hessians[i][0][0])}")
    for j in range(0, len(sorted_hessians[i])):
        idx = np.argsort(sorted_hessians[i][j])
        idx = np.argsort(idx)  
        x = xprev+j*np.ones(len(sorted_hessians[i][j]))
        xprev += j

        #plt.scatter(x, sorted_hessians[i][j], c = idx, s = 3, cmap = "Spectral_r")        

       
xprev = 0
xprevs = [0]
for i in range(0, max_length-1):    
    ith = sorted_hessians[i]
    colors = plt.cm.jet(np.linspace(0,1,i+1))
    for j in range(0, len(ith[0])):
        js = []
        x = [xprev + j for j in range(0, len(ith))]
        for h in ith:
            js.append(h[j])

        plt.plot(x, js, color = colors[j], linewidth = 1)
    xprev += len(ith)-1
    xprevs.append(copy.copy(xprev)) 
    plt.axvline(x = xprev, linewidth = .25, color = "black")


plt.axvline(x = 0, linewidth = .25, color = "black", label = "ADAPT Iteration Demarcations")
plt.title("SVD Spectrum of Jacobian Throughout ADAPT-VQE")
plt.xlabel("Cumulative VQE Iterations")
plt.ylabel("Hessian Eigenvalues")
plt.yscale("log")

plt.show()

'''
os.system(f"grep -A1 \"VQE Iter.\" {fname} > temp.dat")

f = open("temp.dat", "r")
Es = []
for line in f.readlines():
    if "VQE Iter." in line and line.split()[-1] == "0":
        Es.append([])
    if "VQE Iter." not in line and "--" not in line:
        Es[-1].append(float(line.split()[-1]))



for i in range(0, len(Es)):
    plt.axvline(x = i+.5, linewidth = .25, color = "black")
    if i != 0:
        x = []
        E = []
    else:
        x = []
        E = []
    for j in range(0, len(Es[i])):
        x.append(i+.6 + .8*(j)/(len(Es[i])-1))
        E.append(Es[i][j])
    plt.plot(x, E, linewidth = 1, color = "black")
    
plt.axvline(.5, linewidth = .25, color = "black", label = "ADAPT Iteration Demarcations")
plt.title("Energy Throughout ADAPT-VQE")
plt.xlabel("ADAPT/VQE Iterations")
plt.ylabel("Energy (a.u.)")

plt.show()
'''
os.system(f"grep -A1 \"VQE Iter.\" {fname} > temp.dat")

f = open("temp.dat", "r")

Es = []
for line in f.readlines():
    if "VQE" in line and line.split()[-1] == "0":
        Es.append([])
    if "VQE" not in line and "--" not in line:
        Es[-1].append(float(line.split()[-1])-ci)

xprev = 0
Eprev = scf-ci
for i in range(0, len(Es)):
    E = [Eprev]
    x = [xprev]
    for j in range(0, len(Es[i])):
        xprev += 1
        E.append(Es[i][j])
        x.append(xprev)
    Eprev = E[-1]
    plt.plot(x, E, linewidth = 1, color = "black")   
    plt.axvline(x = xprev, linewidth = .25, color = "black")

plt.axvline(0, linewidth = .25, color = "black", label = "ADAPT Iteration Demarcations")
plt.title("Error Throughout ADAPT-VQE")
plt.xlabel("Cumulative VQE Iterations")
plt.ylabel("Error From ED (a.u.)")
plt.yscale("log")

plt.show()


