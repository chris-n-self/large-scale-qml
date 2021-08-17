"""
"""

import time
import logging
from datetime import datetime

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qcoptim.ansatz import AnsatzFromCircuit

from .misc import _dump_output

logger = logging.getLogger(__name__)


#
# Circuit functions
# ------------------
#


def _pretranspile_pqc(
    pqc, instance, transpiler, dump_on_completion=True, dump_directory=None,
):
    """ """
    success_level = None
    ansatz = AnsatzFromCircuit(pqc, strict_transpile=True)

    if transpiler not in [
        'pytket', 'pytket_0', 'pytket_1', 'pytket_2', 'instance'
    ]:
        raise ValueError('transpiler not recognised: '+f'{transpiler}')

    tic = time.time()

    if transpiler == 'instance':
        ansatz.transpiled_circuit(instance, method=transpiler)

    if transpiler in ['pytket', 'pytket_2']:
        try:
            ansatz.transpiled_circuit(
                instance, method='pytket', optimisation_level=2)
            success_level = 2
        except Exception:
            # try again at one optimisation level down
            transpiler = 'pytket_1'
            logger.info('failed to transpile at optimisation_level=2')

    if transpiler == 'pytket_1':
        try:
            ansatz.transpiled_circuit(
                instance, method='pytket', optimisation_level=1)
            success_level = 1
        except Exception:
            # try again at one optimisation level down
            transpiler = 'pytket_1'
            logger.info('failed to transpile at optimisation_level=1')

    if transpiler == 'pytket_0':
        try:
            ansatz.transpiled_circuit(
                instance, method='pytket', optimisation_level=0)
            success_level = 0
        except Exception:
            raise

    toc = time.time()
    logger.info('Transpiling took time : '+f'{toc-tic}'+' seconds')

    # this can be a very expensive operation, so might want to save the result
    if dump_on_completion:
        output = {
            'untranspiled_circuit': pqc,
            'transpiled_circuit': ansatz._transpiled_circuit,
            'instance': instance,
            'time-taken': toc-tic,
            'success_level': success_level,
        }
        # default name
        if dump_directory is None:
            dump_directory = 'tmp/transpiles'
        _dump_output(
            '/'.join([
                dump_directory,
                '-'.join([
                    transpiler,
                    instance.backend.name(),
                    datetime.now().strftime("%y%m%d_%H%M")
                ])
            ]), output
        )

    return ansatz


def _make_natural_angles(n_qubits, depth):
    """
    Generate natural parameter angles

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    depth : int
        Depth of natural PQC

    Returns
    -------
    np.ndarray, shape (depth,n_qubits)
        Natural parameter angles
    """
    ini_angles = np.zeros([depth, n_qubits])
    ini_angles[1:depth:2, :] = 0
    ini_angles[0:depth:2, :] = np.pi/2
    return ini_angles


def _make_circuit_instructions(n_qubits, depth, type_circuit):
    """
    Generate list of single qubit and entangling instructions

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    depth : int
        Depth of natural PQC
    type_circuit : int
        Options:
            0: natural parameterized quantum circuit (NPQC)
            1: NPQC without ring
            2: NPQC ring with additional SWAP and 4 parameters (special case)
            3: YZ CNOT alternating circuit

    Returns
    -------
    nested list of int
        `ini_pauli`, the per layer specification of single qubit gates
    nested list of int or []
        `entangling_gate_index_list`, specification of entangling gates
    """

    if type_circuit in [0, 1, 2]:

        # if type_circuit == 1:
        #     if depth > 8:
        #         raise ValueError(
        #             "For type-1 circuits, only at most depth=8 allowed!"
        #         )

        # define rotations for circuit in each layer, 0: identity, 1:X, 2:Y 3:Z
        ini_pauli = np.zeros([depth, n_qubits], dtype=int)

        # set first and second layer, rest comes later
        ini_pauli[0, :] = 2  # y rotation
        if depth > 1:
            ini_pauli[1, :] = 3  # z rotation

        # construct natural parameterized circuit
        # gives which type of entangling gates at each layer -- first entry is
        # first qubit index, second is second qubit index, third entry is type
        # of entangling gate
        entangling_gate_index_list = [[] for i in range(depth)]
        orderList = []
        for i in range(n_qubits//2):
            if i % 2 == 0:
                orderList.append(i//2)
            else:
                orderList.append((n_qubits-i)//2)

        if n_qubits > 1:
            shiftList = [orderList[0]]
        else:
            shiftList = []
        for i in range(1, n_qubits//2):
            shiftList.append(orderList[i])
            shiftList += shiftList[:-1]

        # this list gives which entangling gates are applied in which layer
        if type_circuit == 0:
            # deep natural PQC, includes non-nearest neighbor gates
            for j in range(min(len(shiftList), int(np.ceil(depth/2))-1)):
                entangling_gate_index_list[1+2*j] = [
                    [2*i, (2*i+1+2*shiftList[j]) % n_qubits, 0]
                    for i in range(n_qubits//2)
                ]
        elif type_circuit == 1:
            # only do 2 entangling layers at max, and only do gates with
            # nearest neighbor and no ring
            for j in range(min(len(shiftList), 3)):
                if j == 0:
                    entangling_gate_index_list[1+2*j] = [
                        [2*i, (2*i+1+2*shiftList[j]) % n_qubits, 0]
                        for i in range(n_qubits//2)
                    ]
                elif (j == 1 or j == 2):
                    # exclude ring gate and gate 0,1 on third entangling layer
                    entangling_gate_index_list[1+2*j] = [
                        [2*i, (2*i+1+2*shiftList[j]) % n_qubits, 0]
                        for i in range(1, n_qubits//2)
                    ]

        elif type_circuit == 2:
            # only do 3 regular entangling layers in a ring topology, then two
            # more phase gates with next-nearst neighbor, which requires one
            # swap. This adds 4 more parameters
            for j in range(min(len(shiftList), 3)):
                entangling_gate_index_list[1+2*j] = [
                    [2*i, (2*i+1+2*shiftList[j]) % n_qubits, 0]
                    for i in range(n_qubits//2)
                ]
            # entangling_gate_index_list[1+2*3]=[[0,n_qubits-1,1],[0,1,0],[n_qubits-1,n_qubits-2,0]]
            # entangling_gate_index_list[1+2*3]=[[0,n_qubits-1,1],[0,1,0],[n_qubits-1,n_qubits-2,0]]
            entangling_gate_index_list[1+2*3] = [
                [n_qubits-1, 1, 0],
                [0, n_qubits-2, 0]
            ]

        for i in range(len(entangling_gate_index_list)-1):
            if len(entangling_gate_index_list[i]) > 0:
                for j in range(len(entangling_gate_index_list[i])):
                    qubit_index = entangling_gate_index_list[i][j][0]
                    ini_pauli[i+1, qubit_index] = 2
                    if i+2 < depth:
                        ini_pauli[i+2, qubit_index] = 3

    elif type_circuit == 3:

        ini_pauli = np.ones([depth, n_qubits], dtype=int)*2

        for i in range(1, depth, 2):
            ini_pauli[i, :] = 3

        if n_qubits % 2 == 0:
            # even qubits ALT circuit needs to get rid of boundary rotations at
            # even entangling layers
            for i in range(4, depth, 4):
                ini_pauli[i, 0] = 0
                ini_pauli[i, -1] = 0
                if i+1 < depth:
                    ini_pauli[i+1, 0] = 0
                    ini_pauli[i+1, -1] = 0
        else:
            # for odd qubits, get rid of boundary either on top or bottom qubit
            for i in range(2, depth, 4):
                ini_pauli[i, -1] = 0
                if i+1 < depth:
                    ini_pauli[i+1, -1] = 0
            for i in range(4, depth, 4):
                ini_pauli[i, 0] = 0
                if i+1 < depth:
                    ini_pauli[i+1, 0] = 0

        # CNOT entangling gates
        entangling_gate_index_list = [[] for i in range(depth)]
        counter = 0
        # third index indicates type of entangling gate
        for k in range(1, depth-1, 2):

            # place entangler every second layer, do not place any at last
            if counter % 2 == 0:
                # even layer
                entangling_gate_index_list[k] = [
                    [2*j, 2*j+1, 1] for j in range(n_qubits//2)
                ]
            else:
                # odd layer
                entangling_gate_index_list[k] = [
                    [2*j+1, 2*j+2, 1] for j in range((n_qubits-1)//2)
                ]
            counter += 1

    else:
        raise ValueError('type_circuit='+f'{type_circuit}'+' not recognised.')

    return ini_pauli, entangling_gate_index_list


def make_natural_parameterised_circuit(
    n_qubits,
    depth,
    n_features,
    type_circuit=0,
    initial_angles='natural',
    random_seed=None,
):
    """
    Make natural parameterised circuit capable of encoding a fixed number of
    features

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    depth : int
        Depth of natural PQC to generate
    n_features : int
        Number of features to encode, equal to number of parameters that will
        be inserted
    type_circuit : int, optional
        When set to 1, circuits only have nearest-neighbour entangling gates
        and do not have periodic boundaries so are more compatible with real
        devices
    initital_angles : {'natural', 'random', 'zeros'}, optional
        Angles to centre feature parameters around
    random_seed : int, optional
        Random seed to use for random initial_angles

    Returns
    -------
    qiskit.QuantumCircuit
        Natural parameterised circuit
    list[qiskit.circuit.Parameter]
        List of parameter objs used in circuit to encode features
    """
    if (n_qubits % 2 == 1) and (type_circuit in [0, 1, 2]):
        raise ValueError(
            'Requested circuit setup, n_qubits='+f'{n_qubits}'
            + ', type_circuit='+f'{type_circuit}'+' not allowed. Circuit types'
            + ' 0, 1, 2 only support even numbers of qubits.'
        )

    # random generator used
    rng = np.random.default_rng(random_seed)

    # define angles for circuit
    if initial_angles == 'natural':
        # note that not all angles are actually used, the ones where
        # ini_pauli=0 are ignored
        ini_angles = _make_natural_angles(n_qubits, depth)
    elif initial_angles == 'random':
        ini_angles = rng.random([depth, n_qubits])*2*np.pi
    elif initial_angles == 'zeros':
        ini_angles = np.zeros([depth, n_qubits])
    else:
        raise ValueError(
            'Invalid option for initial_angles: '+f'{initial_angles}'
            + ', please choose "natural", "random" or "zeros".'
        )

    # get circuit instructions
    ini_pauli, entangling_gate_index_list = _make_circuit_instructions(
        n_qubits, depth, type_circuit)

    # make circuit
    circuit = QuantumCircuit(n_qubits)
    parameters = []
    for j in range(depth):
        for k in range(n_qubits):
            type_pauli = ini_pauli[j][k]
            if type_pauli != 0:

                angle = ini_angles[j][k]
                if len(parameters) < n_features:
                    new_param = Parameter('R'+str(len(parameters)))
                    parameters.append(new_param)
                    angle += new_param

                if type_pauli == 1:
                    circuit.rx(angle, k)
                elif type_pauli == 2:
                    circuit.ry(angle, k)
                elif type_pauli == 3:
                    circuit.rz(angle, k)
                else:
                    raise ValueError(
                        'rotation gate type: '+f'{type_pauli}'
                        + ', not recognised.'
                    )

        if len(entangling_gate_index_list[j]) > 0:
            for gate_indices in entangling_gate_index_list[j]:
                if gate_indices[2] == 0:
                    # entangling gates of nPQC, pi/2 y rotation on control
                    # qubit, followed by CPHASE
                    circuit.ry(np.pi/2, gate_indices[0])
                    circuit.cz(gate_indices[0], gate_indices[1])
                elif gate_indices[2] == 1:
                    # CNOT
                    circuit.cnot(gate_indices[0], gate_indices[1])
                else:
                    raise ValueError(
                        'entangling gate type: '+f'{gate_indices[2]}'
                        + ', not recognised.'
                    )

    if len(parameters) < n_features:
        raise ValueError(
            'Circuit did not have enough gates ('+f'{len(parameters)}'
            + ' params) to encode requested number of'+' features: '
            + f'{n_features}'+'.'
        )

    return circuit, parameters
