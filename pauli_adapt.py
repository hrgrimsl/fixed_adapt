from driver import *
from pyscf_backend import *
from of_translator import *
import numpy as np
import copy
from sys import argv
from pyscf import gto, scf
import math
script, chain_length, bond_distance, hf_file, adapt_file = argv

chain_length = int(chain_length)
bond_distance = float(bond_distance)
print("Doing HF across PES to ensure continuity...")
d = .7
guess = "atom"

print(bond_distance)
while d < bond_distance - .025:
   geom = "H 0 0 0"
   for j in range(1, chain_length):
       geom += f"; H 0 0 {j*d}"
   mol = gto.M(atom = geom, basis = "STO-3G", spin = 0, charge = 0, verbose = False)
   mol.symmetry = False
   mol.max_memory = 8e3
   mol.build()
   mf = scf.RHF(mol)
   mf.chkfile = hf_file
   mf.conv_tol = 1e-12
   mf.verbose = 0
   mf.max_cycle = 1000
   mf.conv_check = True
   mf.init_guess = guess
   hf_energy = mf.kernel()
   if guess != 'read':
       print("r,HF")
   guess = 'read'
   print(f"{d},{hf_energy}")
   d += .05
   assert mf.converged == True


geom = "H 0 0 0"
for i in range(1, chain_length):
    geom += f"; H 0 0 {i*bond_distance}"

print("Geometry:")
print(geom)

E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, "STO-3G", "rhf", chkfile = hf_file, read = True)
N_e = int(np.trace(D))
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
'''
w, v = scipy.sparse.linalg.eigsh(H, k = 3, which = "SA")

for i in range(0, 1):
    print(v[:,i])
    print(v[:,i].T@(S2)@v[:,i])
    idx = np.argsort(abs(v[:,i]))[::-1]
    for j in range(0, 15):
        print(f"{idx[j]} {v[:,i][idx[j]]}")
    print("---")
exit()
'''
E, params, error = xiphos(H, ref, N_e, N_qubits, S2, Sz, Nop, thresh = None, depth = 100, chem_acc = False, subspace_algorithm = "adapt", units = "Eh", pool = "4qubit", out_file = adapt_file, sample_vqe = True, seeds = 125, sample_uccsd = False, dump_dir = 'pauli', rscale = math.pi*.5)
#Following line used for CF, to determine depth to get chemical accuracy
#E, params, error = xiphos(H, ref, N_e, N_qubits, S2, Sz, Nop, thresh = None, chem_acc = True, subspace_algorithm = "adapt", units = "Eh", pool = "uccsd", out_file = adapt_file)

print(f"HF Energy:           {hf_energy:20.16f}")
print(f"ADAPT Energy:        {E:20.16f}")
print(f"FCI Energy:          {E-error:20.16f}")


