import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

df_u = pd.read_csv('uccsd.csv')
df_t = pd.read_csv('ucc_new.csv')
df_r = pd.read_csv('recycled.csv')

chem = 1/627.5094740631
x = np.array([i+1 for i in range(0, np.amax(df_u['Operators'].to_numpy()))])
vanilla_x = df_r['Operators']
vanilla_err = df_r['Error']

points = []
for i in x:
    points.append(np.array([df_u["Random E"][j] for j in range(0, len(df_u["Random E"])) if df_u["Operators"][j] == i]))
u_stds = [np.std(pt) for pt in points]

points = []
for i in x:
    points.append(np.array([df_t["E"][j] for j in range(0, len(df_t["E"])) if df_t["Operators"][j] == i]))
t_stds = [np.std(pt) for pt in points]

plt.xlabel("Operators in Ansatz")
plt.ylabel("Standard Deviation in Energy (a.u.)")
plt.title("H6, 3$\AA$ Bond Length")
plt.plot(x, t_stds, label = 'ADAPT ansatz BFGS solutions')
plt.plot(x, u_stds, label = 'Untrotterized ADAPT ansatz BFGS solutions')
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
