#Driver for new ADAPT project with Bryan
import system_methods as sm
import computational_tools as ct
import scipy
import copy
import openfermion as of
import numpy as np

#Globals
Eh = 627.5094740631

def qubit_adapt(H, ref, N_e, N_qubits, S2, thresh = 1e-3, out_file = 'out.dat', factor = Eh):
    f = open(out_file, 'w')
    system = sm.system_data(H, ref, N_e, N_qubits)
    print("Done! Generating qubit pool...")
    pool = []
    for i in range(1, 5):
        pool += system.k_qubit_pool(i)
    print("FCI Energy:")
    print(system.ci_energy)
    print("SCF Energy:")
    print(system.hf_energy)
    print("Done! Performing ADAPT calculation...")

    cur_state = system.ref
    ansatz = []
    params = []
    Done = False
    iteration = 0
    print('\n')
    print('{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|'.format('Iteration', 'Energy (kcal/mol)', 'Error (kcal/mol)', 'Last del (a.u.)', 'Newest Operator', '<S^2>'))
    while Done == False:
        iteration += 1
        max_grad = 0
        new_op = None
        for i in range(len(pool)):
            op = copy.copy(pool[i])
            grad = 2*abs(cur_state.T.dot(system.H).dot(op).dot(cur_state)[0,0].real)
            if grad > max_grad:
                max_grad = copy.copy(grad)
                new_op = copy.copy(op)
                new_idx = copy.copy(i)
        if max_grad < thresh:
            Done = True 
        else:
            ansatz = [new_op] + ansatz
            params = [0] + params
            energy, params = ct.vqe(system.H, ansatz, system.ref, params)
            cur_state = copy.copy(system.ref)
            for i in reversed(range(0, len(params))): 
                cur_state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], cur_state)
            S2_state = cur_state.T.dot(S2).dot(cur_state)[0,0].real
            print('-'*105)
            print('{:<20}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20}|{:<20}|'.format(iteration, factor*energy, factor*(energy-system.ci_energy), max_grad, system.pool[new_idx], S2_state))
            #print(str(factor*energy)+' '+str(factor*(energy-system.ci_energy)))
            f.write(system.pool[new_idx]+'\n')
    print('\n')
    return energy, params

def fixed_adapt(H, ref, N_e, N_qubits, params, thresh = 1e-3, in_file = 'out.dat', factor = Eh):
    system = sm.system_data(H, ref, N_e, N_qubits)
    #print("Performing VQE on pre-specified ansatz...\n")
    ansatz = []

    f = open(in_file, 'r')
    count = 0
    for line in f.readlines():
        ansatz = [of.get_sparse_operator(1j * of.ops.QubitOperator(line), system.N_qubits)] + ansatz


    E, params = ct.vqe(system.H, ansatz, system.ref, params)
    #print("VQE energy (kcal/mol):")

    print(str(E*factor)+' '+str(factor*(E-system.ci_energy)))
    #print("Error (kcal/mol):")
    #print(factor*(E-system.ci_energy))
    return E, params
if __name__ == "__main__":
    geometry = [('H', (0,0,0)), ('H', (0,0,.74))]
    basis = 'sto-3g'
    multiplicity = 1
    system = sm.system_data(geometry, basis, multiplicity)
    pool = system.full_qubit_pool()

