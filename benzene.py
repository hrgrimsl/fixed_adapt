from pyscf_backend import *
from of_translator import *

#Geometry in angstroms, optimized with Psi4.
#(B3LYP DFT calculation with a 6-311G(d,p) basis set)
geometry = """
    C           -1.200649405854     0.707888282790     0.000000000000
    C           -1.213422696760    -0.685836804691     0.000000000000
    C           -0.012744802332    -1.393703415845     0.000000000000
    C            1.200649405854    -0.707888282790     0.000000000000
    C            1.213422696760     0.685836804691     0.000000000000
    C            0.012744802332     1.393703415845     0.000000000000
    H            0.022631937859     2.478034113756     0.000000000000
    H           -2.134744664412     1.258650527007     0.000000000000
    H            2.134744664412    -1.258650527007     0.000000000000
    H            2.157428775620     1.219431010435     0.000000000000
    H           -2.157428775620    -1.219431010435     0.000000000000
    H           -0.022631937859    -2.478034113756     0.000000000000
"""

basis = "sto-3g"

#I only trust RHF right now- I can debug open-shell refs if you need them though.
reference = "rhf"

#Number of frozen electrons.  Probably an even number.
frozen_core = 36

#Number of frozen virtual orbitals.
frozen_vir = 24


'''
In this case you have reference:

|111100000000>

becoming

Frozen |11> + Active |11000000> + Frozen |00>
= |core> + |active> + |vir>

where you've frozen the 2 electrons in the lowest energy orbitals,
and simply removed the highest 2 spin-orbitals from consideration.
For RHF, this is equivalent to freezing the highest and lowest spatial orbitals.
'''

#Use PySCF to compute spatial orbital integrals, then put them into inefficient spin-orbital tensors that OpenFermion can read
_0body_H, _1body_H, _2body_H, _1rdm, hf_energy = get_integrals(geometry, basis, reference, frozen_core = frozen_core, frozen_vir = frozen_vir)

#_0body_H is the nuclear repulsion + <core|H|core>
#_1body_H is the 1-body part of <active|H|active> + <core|H|active> 
#_2body_H is the 2-body part of <core|H|core>
#_1rdm is the MO basis rdm.
#hf_energy is <ref|H|ref> and doesn't care about your orbital freezing
#These are all in the weird, proprietary format of pyscf's tensor construction

#Get number of electrons:
N_e = int(np.trace(_1rdm))

#Use OpenFermion to build sparse matrix representations of JW-transformed operators:
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(_0body_H, _1body_H, _2body_H, N_e)

#H is the Hamiltonian, ref is the HF reference in the active space.  S2, S_z, and N_op are the S^2, S_z, and number operators if you happen to want them.

ci_E, ci_v = np.linalg.eigh(H.todense())
ci = ci_E[0]
print(scipy.sparse.csc_matrix(ci_v[:,0]))
print(f"Problem reduced to {N_qubits} qubits.")
print(f"SCF energy:                {hf_energy:20.16f}")
print(f"Active space FCI energy:   {ci:20.16f}")

