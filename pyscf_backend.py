from pyscf import *
import numpy as np
from opt_einsum import contract

def get_integrals(geometry, basis, reference, charge = 0, spin = 0):

    mol = gto.M(atom = geometry, basis = basis)
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.max_memory = 8e3
    mol.build()
    if reference == 'rhf':
        mf = scf.RHF(mol)
    elif reference == 'rohf':
        mf = scf.ROHF(mol)
    else:
        print('Reference not supported.')
    mf.max_cycle = 100
    mf.verbose = 0
    hf_energy = mf.kernel()
    mo_occ = mf.mo_occ
    C = mf.mo_coeff
    mo_a = np.zeros(len(mo_occ))
    mo_b = np.zeros(len(mo_occ))
    for i in range(0, len(mo_occ)):
        if mo_occ[i] > 0:
            mo_a[i] = 1
        if mo_occ[i] > 1:
            mo_b[i] = 1


    Da = np.diag(mo_a)
    Db = np.diag(mo_b)
    N_e = np.trace(Da) + np.trace(Db)
    S = mol.intor('int1e_ovlp_sph')
    E_nuc = mol.energy_nuc()
    H = mol.intor('int1e_nuc_sph') + mol.intor('int1e_kin_sph')
    I = mol.intor('int2e_sph')
    H = C.T.dot(H).dot(C)
    I = contract('pqrs,pi,qj,rk,sl->ijkl', I, C, C, C, C)
    Ja = .5*contract('pqrs,rs->pq', I, Da)
    Jb = .5*contract('pqrs,rs->pq', I, Db)
    Ka = .5*contract('psrq,rs->pq', I, Da)
    Kb = .5*contract('psrq,rs->pq', I, Db)
    Fa = .5*H + Ja + Jb - Ka
    Fb = .5*H + Ja + Jb - Kb
    hf_energy = E_nuc + contract('pq,pq', .5*H + Fa, Da) + contract('pq,pq', .5*H + Fb, Db)

    idensor = np.array([[[[1,0],[0,1]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[1,0],[0,1]]]])
    I = np.kron(I, idensor)
    H = np.kron(H, np.eye(2))
    D = np.kron(Da, np.array([[1,0],[0,0]]))+np.kron(Db, np.array([[0,0],[0,1]]))
    C = np.kron(C, np.array([[1,0],[0,0]]))+np.kron(C, np.array([[0,1],[0,0]]))

    #Build Coulomb/Exchange in MO basis
    J = contract('pqrs,rs->pq', I, D)
    K = contract('psrq,rs->pq', I, D)
    manual_hf = .5*contract('pq,pq->', (2*H + J - K), D) + E_nuc
    assert(abs(manual_hf-hf_energy)<=1e-9)
    return E_nuc, H, I, D, C

def freeze_core(E_nuc, H, I, D, N_c):
    D_core = D[:N_c,:N_c]
    J_core = contract('pqrs,rs->pq', I[:N_c, :N_c, :N_c, :N_c], D_core) 
    K_core = contract('psrq,rs->pq', I[:N_c, :N_c, :N_c, :N_c], D_core)
    zero_body = E_nuc + .5*contract('pq,pq->', (2*H[:N_c,:N_c] + J_core - K_core), D_core)
    J_mix = contract('pqrs,rs->pq', I[N_c:, N_c:, :N_c, :N_c], D_core)
    K_mix = contract('psrq,rs->pq', I[N_c:, :N_c, :N_c, N_c:], D_core)
    one_body = H[N_c:, N_c:] + J_mix - K_mix   
    two_body = I[N_c:, N_c:, N_c:, N_c:] 
    return zero_body, one_body, two_body, D[N_c:, N_c:]

def rotate(H, I, R, rotate_rdm = False):
    #Typically you'll do 2 rotations- one to rotate into the AO basis, and another to rotate into the new basis
    H = contract('pq,pi,qj->ij', H, R, R)
    I = contract('pqrs,pi,qj,rk,sl->ijkl', I, R, R, R, R)
    return H, I
