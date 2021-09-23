import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

df = pd.read_csv('diss_sampled_adapt.csv')
chem = 1/627.5094740631


#Count unique solutions
solns = []
x = [i for i in range(1, np.amax(df_t["Ops"]+1))]
sol = []
for i in x:
    sol.append([])
    solns.append({})
    for j in range(0, len(df_t["Ops"])):
        if df_t["Ops"][j] == i:
           sol[-1].append(df_t["E"][j])
           ignore = False
           for k in solns[-1].keys():
               if abs(df_t["E"][j] - k) < 1e-8:
                   solns[-1][k] += 1
                   ignore = True
           if ignore == False:
               if i == 2:
                   print(df_t["E"])
               solns[-1][df_t["E"][j]] = 1
sols = []
for i in range(0, len(solns)):
    sols.append(len(solns[i].keys())/12)
best = [min(i) - df_t["Exact"][0] for i in sol] 
devs = [np.std(np.array(sol[i])) for i in range(0, len(sol))]
print(devs)
plt.xlabel("Operators in Ansatz")
plt.ylabel("Error from FCI (a.u.)")
plt.title("H6, 3$\AA$ Bond Length")
plt.scatter(df_t["Ops"], df_t["E"]-df_t["Exact"], label = 'Solutions to Different BFGS Initializations', alpha = .2, s = 1)
#plt.plot(x, best, label = 'Best energy')
print(x)
print(best)
plt.yscale("symlog", linthresh = 1e-8)
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
    for t in range(0, len(df_t['Random E'])):
        if df_t["Operators"][t] != i+1:
            continue
        if df_t['Random E'][t] < min_t[i]:
            min_t[i] = copy.deepcopy(df_t['Random E'][t])
        if df_t['Recycled E'][t] < min_t[i]:
            min_t[i] = copy.deepcopy(df_t['Recycled E'][t])

plt.yscale('symlog', linthresh = 1e-8)
plt.scatter(df_t["ansatz"], df_t["dumb_E"]-df_t["dumb_CI"], alpha = .05, label = "E$_0$ From Random initializations of Trotterized UCCSD")
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
