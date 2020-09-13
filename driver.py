#Driver for new ADAPT project with Bryan
import system_methods as sm
import computational_tools as ct
import scipy
import copy

def qubit_adapt(geometry, basis, multiplicity, thresh = 1e-8):
    print("Building Hamiltonian...")
    system = sm.system_data(geometry, basis, multiplicity)
    print("Done! Generating qubit pool...")
    pool = system.full_qubit_pool()
    print("Done! Performing ADAPT calculation...")
    cur_state = system.ref
    ansatz = []
    params = []
    Done = False
    iteration = 0
    print('\n')
    print('{:<20}|{:<20}|{:<20}|{:<20}|'.format('Iteration', 'Energy', 'Last Gradient Norm', 'Newest Operator'))
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
            print('-'*84)
            print('{:<20}|{:<20.12}|{:<20.12}|{:<20}|'.format(iteration, energy, max_grad, system.pool[new_idx]))

    print('\n')


    print("Exact FCI energy:")
    print(system.mol.fci_energy)
if __name__ == "__main__":
    geometry = [('H', (0,0,0)), ('H', (0,0,.74))]
    basis = 'sto-3g'
    multiplicity = 1
    system = sm.system_data(geometry, basis, multiplicity)
    pool = system.full_qubit_pool()

