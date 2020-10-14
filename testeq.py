from driver import *
from p4n_backend import *
from of_translator import *
import numpy as np
import copy
geometry = """
0 1
H 0 0 0
Li 0 0 1.6
symmetry c1
"""

N_c = 2

print("Real ADAPT:")
E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
N_e = int(np.trace(D))
E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
N_e -= N_c
H, ref, N_qubits, S2 = of_from_arrays(E_nuc, H, I, N_e, S_squared = 0, S_z = 0)

#E, zero_params = qubit_adapt(H, ref, N_e, N_qubits, S2, factor = Eh, thresh = 5e-5, out_file = 'eq2storage.dat')

zero_params = np.zeros(41)
params1 = copy.copy(zero_params)
params2 = copy.copy(zero_params)
print("Fixed ADAPT:")
print("Scanning backward...")
for r in reversed(range(10,17)): 
    geometry = """
    0 1
    H 0 0 0
    """
    geometry += "Li 0 0 "+str(.1*r)+"\nsymmetry c1"
    N_c = 2

    E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
    N_e = int(np.trace(D))

    #H, I = rotate(H, I, R, rotate_rdm = False)

    E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
    N_e -= N_c

    H, ref, N_qubits, S2 = of_from_arrays(E_nuc, H, I, N_e, S_squared = 0, S_z = 0)
    E, params1 = fixed_adapt(H, ref, N_e, N_qubits, params1, factor = Eh, in_file = 'eq2storage.dat')
print("Scanning forward...")
for r in range(17,46): 
    geometry = """
    0 1
    H 0 0 0
    """
    geometry += "Li 0 0 "+str(.1*r)+"\nsymmetry c1"
    N_c = 2

    E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
    N_e = int(np.trace(D))

    #H, I = rotate(H, I, R, rotate_rdm = False)

    E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
    N_e -= N_c

    H, ref, N_qubits, S2 = of_from_arrays(E_nuc, H, I, N_e, S_squared = 0, S_z = 0)
    E, params2 = fixed_adapt(H, ref, N_e, N_qubits, params2, factor = Eh, in_file = 'eq2storage.dat')




