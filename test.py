from driver import *

geometry = [('H', (0, 0, .26)), ('He', (0, 0, 1)), ('H', (0, 0, 1.74))]
#geometry = [('H', (0, 0, .26)), ('H', (0, 0, 1.74))]
basis = 'sto-3g'
multiplicity = 1

qubit_adapt(geometry, basis, multiplicity, thresh = 1e-3)
