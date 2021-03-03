from driver import *
from pyscf_backend import *
from of_translator import *
import numpy as np
geometry = 'H 0 0 0; H 0 0 2; H 0 0 4; H 0 0 6; H 0 0 8; H 0 0 10'
N_c = 0


E_nuc, H, I, D, C, hf_energy= get_integrals(geometry, "sto-3g", "rhf")
N_e = int(np.trace(D))
print(E_nuc)
np.save('/home/hrgrimsl/terror/h', H)
np.save('/home/hrgrimsl/terror/I', I)
grad = hess(params)
hess = hess(params, H, ansatz, ref)

exit()

