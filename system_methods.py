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

    def recursive_qubit_op(self, op, qubit_index):
        if qubit_index == self.n_qubits-1:
            return [op, op + ' X' + str(qubit_index), op + ' Y' + str(qubit_index), op + ' Z' + str(qubit_index)]
        else:
            return self.recursive_qubit_op(op, qubit_index+1) + self.recursive_qubit_op(op + ' X' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Y' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Z' + str(qubit_index), qubit_index+1)

        
    def full_qubit_pool(self):
        pool = []
        pool += self.recursive_qubit_op("", 0)
        assert(len(pool) == 4**self.n_qubits)
        self.pool = [i for i in pool if len(re.findall("Y", i))%2 == 1]
        return [of.get_sparse_operator(1j * of.ops.QubitOperator(i), self.n_qubits) for i in pool]

            
