import psi4
import numpy as np
from opt_einsum import contract

def get_integrals(geometry, basis, reference):
    mol = psi4.geometry(geometry)
    psi4.core.be_quiet()
    psi4.set_options({'basis': basis,
                      'reference': reference,
                      'scf_type': 'pk',
                      'e_convergence': 1e-8,
                      'diis_max_vecs': 100,
                      'memory': '6GB'})
    hf_energy, wfn = psi4.energy('scf', return_wfn = True)
    mints = psi4.core.MintsHelper(wfn.basisset())
    S = np.array(mints.ao_overlap())
    M = S.shape[0]*2
    N_e = wfn.nalpha()+wfn.nbeta()
    Ca = wfn.Ca().to_array()
    Cb = wfn.Cb().to_array()
    C = np.kron(Ca, np.array([[1,0],[0,0]])) + np.kron(Cb, np.array([[0,0],[0,1]]))  
    # Build ERI Tensor (chemist's notation)
    I = np.asarray(mints.ao_eri())
    idensor = np.array([[[[1,0],[0,1]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[1,0],[0,1]]]])
    I = np.kron(I, idensor)
    I = contract('pqrs,pi,qj,rk,sl->ijkl', I, C, C, C, C)
    # Build Core Hamiltonian
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    E_nuc = mol.nuclear_repulsion_energy()
    H = T + V 
    H = np.kron(H, np.eye(2))
    #H = C.T.dot(H).dot(C)
    H = contract('pq,pi,qj->ij', H, C, C)
    # Build 1RDM
    D = np.zeros((M, M)) 
    D[:N_e, :N_e] = np.eye(N_e)
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
