#Gives molecule class w/ methods for deriving pools, etc.
import openfermion as of
import openfermionpsi4 as ofp
import scipy
import re
class system_data:
    def __init__(self, geometry, basis, multiplicity):
        self.mol = of.MolecularData(geometry, basis, multiplicity)
        self.mol = ofp.run_psi4(self.mol, run_scf = True, run_fci = True)
        self.n_qubits = self.mol.n_qubits
        self.H = of.transforms.get_sparse_operator(self.mol.get_molecular_hamiltonian())
        self.ref = scipy.sparse.csc_matrix(of.jw_configuration_state(list(range(0,self.mol.n_electrons)), self.n_qubits)).transpose()
        self.pool = []
    def recursive_qubit_op(self, op, qubit_index):
        if qubit_index == self.n_qubits-1:
            return [op, op + ' X' + str(qubit_index), op + ' Y' + str(qubit_index), op + ' Z' + str(qubit_index)]
        else:
            return self.recursive_qubit_op(op, qubit_index+1) + self.recursive_qubit_op(op + ' X' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Y' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Z' + str(qubit_index), qubit_index+1)
        
    def choose_next(self, set_of_lists, cur_list, k):
        if len(cur_list) == k:
            set_of_lists += [cur_list]
        else:
            if len(cur_list) == 0:
                floor = 0
            else:
                floor = max(cur_list)+1
            for i in range(floor, self.n_qubits):
                self.choose_next(set_of_lists, cur_list+[i], k)

    def choose_paulis(self, paulis, sub_list, k):
        if len(sub_list) == k:
            paulis += [sub_list]
        else:
            for let in ['X', 'Y', 'Z']:
                self.choose_paulis(paulis, sub_list + [let], k)
                        

    def full_qubit_pool(self):
        pool = []
        pool += self.recursive_qubit_op("", 0)
        assert(len(pool) == 4**self.n_qubits)
        self.pool += [i for i in pool if len(re.findall("Y", i))%2 == 1]
        return [of.get_sparse_operator(1j * of.ops.QubitOperator(i), self.n_qubits) for i in self.pool]

    def k_qubit_pool(self, k):
        indices = []
        self.choose_next(indices, [], k)
        paulis = []
        self.choose_paulis(paulis, [], k)
        pool = []
        for i in indices:
            for j in paulis:
                string = str(j[0])+str(i[0])
                for l in range(1, len(i)):
                    string += " "+str(j[l])+str(i[l])
                pool.append(string)
        pool = [i for i in pool if len(re.findall("Y", i))%2 == 1]
        self.pool += pool 
        return [of.get_sparse_operator(1j * of.ops.QubitOperator(i), self.n_qubits) for i in pool]

        
            
