from driver import *

basis = 'sto-3g'
multiplicity = 1


geometry = [('H', (0, 0, 0)), ('Li', (0, 0, 3.5))]
qubit_adapt(geometry, basis, multiplicity, thresh = 1e-3)

geometry = [('H', (0, 0, 0)), ('Li', (0, 0, 1.5))]
fixed_adapt(geometry, basis, multiplicity)

geometry = [('H', (0, 0, 0)), ('Li', (0, 0, 1))]
fixed_adapt(geometry, basis, multiplicity)

geometry = [('H', (0, 0, 0)), ('Li', (0, 0, 1))]
qubit_adapt(geometry, basis, multiplicity, thresh = 1e-3)

geometry = [('H', (0, 0, 0)), ('Li', (0, 0, 1.5))]
fixed_adapt(geometry, basis, multiplicity)

geometry = [('H', (0, 0, 0)), ('Li', (0, 0, 3))]
fixed_adapt(geometry, basis, multiplicity)
