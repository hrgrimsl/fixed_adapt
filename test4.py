from driver import *
from pyscf_backend import *
from of_translator import *
import numpy as np
import copy


print("Spin-adapted SD")
geometry = 'H 0 0 0; Li 0 0 1.6'
N_c = 0
E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
N_e = int(np.trace(D))
E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
N_e -= N_c
H, ref, N_qubits, S2 = of_from_arrays(E_nuc, H, I, N_e)
E, zero_params = uccsd_adapt(H, ref, N_e, N_qubits, S2, factor = Eh, thresh = 1e-3, spin_adapt =  True, out_file = 'misc', depth = 30)
print("Fixed ansatze (sweep from left)")
params = copy.copy(zero_params)
exit()
for i in range(10, 46):
    geometry = 'Li 0 0 0; H 0 0 '+str(i*.1)
    N_c = 0
    E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
    N_e = int(np.trace(D))
    E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
    N_e -= N_c
    H, ref, N_qubits, S2 = of_from_arrays(E_nuc, H, I, N_e)
    E, params = fixed_fermionic(H, ref, N_e, N_qubits, params, factor = Eh, in_file = 'eq')
print("Fixed ansatze (sweep from right)")
params = copy.copy(zero_params)
for i in reversed(range(10, 46)):
    geometry = 'Li 0 0 0; H 0 0 '+str(i*.1)
    N_c = 0
    E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
    N_e = int(np.trace(D))
    E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
    N_e -= N_c
    H, ref, N_qubits, S2 = of_from_arrays(E_nuc, H, I, N_e)
    E, params = fixed_fermionic(H, ref, N_e, N_qubits, params, factor = Eh, in_file = 'eq')
    

'''
E, zero_params = uccgsd_adapt(H, ref, N_e, N_qubits, S2, factor = Eh, thresh = 5e-5, spin_adapt =  False, out_file = 'out.dat')
print("Fixed ADAPT:")
E, params1 = fixed_fermionic(H, ref, N_e, N_qubits, zero_params, factor = Eh)
E, zero_params = uccsd_adapt(H, ref, N_e, N_qubits, S2, factor = Eh, thresh = 5e-5, spin_adapt =  True, out_file = 'out.dat')
print("Fixed ADAPT:")
E, params1 = fixed_fermionic(H, ref, N_e, N_qubits, zero_params, factor = Eh)

E, zero_params = qubit_adapt(H, ref, N_e, N_qubits, S2, factor = Eh, thresh = 1e-2)
print("Fixed ADAPT:")
E, params1 = fixed_adapt(H, ref, N_e, N_qubits, zero_params, factor = Eh)

print("Fixed ADAPT:")
'''



