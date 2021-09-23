import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

df = pd.read_csv('more_pauli.csv')
chem = 1/627.5094740631


#Count unique solutions
solns = []
x = [i for i in range(1, np.amax(df["Ops"]+1))]
recycled_Es = []
adapt_Es = []
random_Es = []
random_E0s = []
for i in x:
    random_Es.append([])
    random_E0s.append([])
    for j in range(0, len(df["Ops"])):
        if df["Ops"][j] == i:
           if df["Initialization"][j] == 0:
               recycled_Es.append(df["E"][j])
           elif df["Initialization"][j] == 1:
               adapt_Es.append(df["E"][j])
           else:
               random_Es[-1].append(df["E"][j])
               random_E0s[-1].append(df["E0"][j])

ci = df["Exact"][0]

plt.yscale("symlog", linthresh = 1e-10)
plt.plot(x, np.array(adapt_Es) - ci, color = "black", marker = '^', label = "Optimization that Mimics ADAPT")
plt.plot(x, np.array(recycled_Es) - ci, color = "black", label = "BFGS Using Recycled Parameters") 


idx = np.argsort(random_Es[-1])
idx = np.argsort(idx)

plt.scatter(x[-1]*np.ones(len(random_Es[-1])), np.array(random_Es[-1]) - ci, label = "BFGS Using Random Initializations", alpha = .3, c = idx, cmap = "Spectral_r") 

#plt.scatter(x[-1]*np.ones(len(random_Es[-1])), np.array(random_E0s[-1]) - ci, label = "Random BFGS Initializations", alpha = .2, c = np.array([random_E0s[-1][i] for i in range(0, len(random_Es[-1]))]), cmap = "Spectral") 

for i in range(0, len(x)-1):
    idx = np.argsort(random_Es[i])
    idx = np.argsort(idx)

    plt.scatter(x[i]*np.ones(len(random_Es[i])), np.array(random_Es[i]) - ci, alpha = .3, c = np.array(idx), cmap = "Spectral_r") 
    #plt.scatter(x[i]*np.ones(len(random_Es[i])), np.array(random_E0s[i]) - ci, alpha = .2, c = np.array([random_E0s[-1][j]-ci for j in range(0, len(random_Es[i]))]), cmap = "Spectral") 

                


plt.xlabel("Operators in Ansatz")
plt.ylabel("Error from Exact Diagonalization (a.u.)")
plt.title("ADAPT-VQE on H6, 3$\AA$ Bond Length, 4-qubit Pool")
plt.legend()
plt.show()
exit()
for i in range(0, len(solns)):
    sols.append(len(solns[i].keys())/12)
best = [min(i) - df["Exact"][0] for i in sol] 
devs = [np.std(np.array(sol[i])) for i in range(0, len(sol))]
print(devs)
plt.xlabel("Operators in Ansatz")
plt.ylabel("Error from FCI (a.u.)")
plt.title("H6, 3$\AA$ Bond Length")
plt.scatter(df["Ops"], df["E"]-df["Exact"], label = 'Solutions to Different BFGS Initializations', alpha = .2, s = 1)
#plt.plot(x, best, label = 'Best energy')
print(x)
print(best)

plt.legend()
plt.show()

exit()
plt.xlabel("Operators in Ansatz")
plt.ylabel("Number of Unique Solutions")
plt.title("H6, 3$\AA$ Bond Length")
plt.plot(x, sols, label = 'Uniqueness Quotient (0 = No solutions unique, 1 = All solutions unique')
plt.axhline(y=0, color = 'black', label = "All 126 solutions identical")
plt.axhline(y=1, color = 'red', label = "All 126 solutions unique")
#plt.yscale("symlog", linthresh = 1e-8)
plt.legend()
plt.show()

plt.xlabel("Operators in Ansatz")
plt.ylabel("Standard Deviation")
plt.title("H6, 3$\AA$ Bond Length")
plt.plot(x, devs, label = 'Standard Deviation')

#plt.yscale("symlog", linthresh = 1e-8)
plt.legend()
plt.show()



min_u = np.zeros(len(x))
min_t = np.zeros(len(x))

for i in range(0, len(x)):
    for u in range(0, len(df_u['Random E'])):
        if df_u["Operators"][u] != i+1:
            continue
        if df_u['Random E'][u] < min_u[i]:
            min_u[i] = copy.deepcopy(df_u['Random E'][u])
        if df_u['Recycled E'][u] < min_u[i]:
            min_u[i] = copy.deepcopy(df_u['Recycled E'][u])
               

for i in range(0, len(x)):
    for t in range(0, len(df['Random E'])):
        if df["Operators"][t] != i+1:
            continue
        if df['Random E'][t] < min_t[i]:
            min_t[i] = copy.deepcopy(df['Random E'][t])
        if df['Recycled E'][t] < min_t[i]:
            min_t[i] = copy.deepcopy(df['Recycled E'][t])

plt.yscale('symlog', linthresh = 1e-8)
plt.scatter(df["ansatz"], df["dumb_E"]-df["dumb_CI"], alpha = .05, label = "E$_0$ From Random initializations of Trotterized UCCSD")
plt.plot(x, min_t-df_u["Exact"][0], label = "Best Trotterized UCCSD Energy")
plt.scatter(df_u["Operators"], df_u["Random E"]-df_u["Exact"], alpha = .05, label = "E$_0$ From Random initializations of Traditional UCCSD")
plt.plot(x, min_u-df_u["Exact"][0], label = "Best Traditional UCCSD Energy")
plt.plot(vanilla_x, vanilla_err, label = "Vanilla ADAPT (No random BFGS initializations.)")
plt.plot(x, chem*np.ones(len(x)), label = "1 kcal/mol")

plt.xlabel("Operators in Ansatz")
plt.ylabel("Error From FCI (a.u.)")
plt.title("Role of Trotterization in Convexity")

plt.legend()
plt.show()
