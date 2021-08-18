"""
"""

import copy
import logging
import numpy as np
from tqdm import tqdm

from qiskit.circuit import Parameter
from qiskit.ignis.mitigation import tensored_meas_cal

from qcoptim.cost.crossfidelity import (
    CrossFidelity,
)
from qcoptim.utilities import (
    make_quantum_instance,
    simplify_rotation_angles,
    zero_rotation_angles,
)

from .utils import (
    _dump_output,
    _get_data_vars,
    _pretranspile_pqc,
    make_natural_parameterised_circuit,
)
from .utils.circuits import _make_circuit_instructions

logger = logging.getLogger(__name__)


def _get_quantum_instances(
    backend_name,
    nb_shots,
    backend_options,
    initial_layout,
    simulate_ibmq,
    noise_model,
    seed_simulator,
    hub,
    group,
    project,
):
    """ """
    exe_instance = make_quantum_instance(
        backend_name,
        measurement_error_mitigation=False,
        nb_shots=nb_shots,
        cals_matrix_refresh_period=0,
        backend_options=backend_options,
        initial_layout=initial_layout,
        simulate_ibmq=simulate_ibmq,
        noise_model=noise_model,
        seed_simulator=seed_simulator,
        hub=hub,
        group=group,
        project=project,
    )
    transpile_instance = exe_instance
    # if simulating an IBMQ device make a non-sim instance for transpiling
    if simulate_ibmq:
        transpile_instance = make_quantum_instance(
            backend_name,
            backend_options=backend_options,
            initial_layout=initial_layout,
            simulate_ibmq=False,
            hub=hub,
            group=group,
            project=project,
        )
    return exe_instance, transpile_instance


def _dump_ibmq_crossfid_results(
    results,
    transpiler_map,
    transpiled_circuit,
    meas_err_mit_assets,
    all_args,
    results_name,
):
    """ """
    output = {
        'fn_args': all_args,
        'qiskit-results': results,
        'transpiler_map': transpiler_map,
        'meas_err_mit_assets': meas_err_mit_assets,
    }

    # be quite careful about this step because it would be devastating if this
    # made us lose some device results
    try:
        output['transpiled-circuit-example-qasm'] = simplify_rotation_angles(
                transpiled_circuit.bind_parameters(
                    np.ones(len(transpiled_circuit.parameters))
                )
            ).qasm()
    except Exception:
        output['transpiled-circuit-example-qasm'] = zero_rotation_angles(
                transpiled_circuit.bind_parameters(
                    np.ones(len(transpiled_circuit.parameters))
                )
            ).qasm()

    _dump_output(results_name, output)


def _make_random_angles(
    type_circuit,
    n_repeat,
    depth,
    n_qubits,
):
    """ """

    ini_pauli, _ = _make_circuit_instructions(
        n_qubits, depth, type_circuit)

    rng2 = np.random.default_rng(2)
    rng3 = np.random.default_rng(3)
    shift_angles = 4  # by how much maximally to randomly shift parameters

    if type_circuit == 1:
        euclidean_ini_angles = np.ones([n_repeat, depth, n_qubits])*np.pi/2
        euclidean_ini_angles[:, 1:depth:2, :] = 0  # np.pi/2

        unshifted_ini_angles = np.array(euclidean_ini_angles)
        random_vector_normed = np.zeros(np.shape(euclidean_ini_angles))
        for k in range(n_repeat):
            random_vector_normed[k] = (
                (2*rng3.random(np.shape(euclidean_ini_angles[k]))-1)
                * (ini_pauli != 0)
            )

        # shift values, takes maximum value ini_shift_angles, take sqrt to
        # sample equally over full sphere
        shift = rng3.random(n_repeat)*shift_angles

        shift[0] = 0  # set first shift to zero for qfi
        for i in range(n_repeat):
            random_vector_normed[i] = (
                random_vector_normed[i]
                / np.sqrt(np.sum(np.abs(random_vector_normed[i])**2))
            )
        ini_angles = (
            unshifted_ini_angles
            + np.array([
                random_vector_normed[i]*shift[i] for i in range(n_repeat)
            ])
        )

    if type_circuit == 3:

        ini_angles = np.zeros([n_repeat, depth, n_qubits])
        rand_angles = rng2.random([depth, n_qubits])*2*np.pi
        for i in range(n_repeat):
            ini_angles[i] = rand_angles

        unshifted_ini_angles = np.array(ini_angles)
        random_vector_normed = np.zeros(np.shape(ini_angles))
        for k in range(n_repeat):
            random_vector_normed[k] = (
                (2*rng3.random(np.shape(ini_angles[k]))-1)
                * (ini_pauli != 0)
            )

        # unit vector of length 1, L2 norm is correct due to
        # d_b=1-Fidelity(dx)=1/2*dx F dx, where F fisher metric, which is
        # 4*Fubini (fubini is calculated here) e.g. for flat norm ,
        # ini_shift_angles=2*np.sqrt(0.01) should give 1-0.01 fidelity. Note
        # that factor 2 due to bures metric and 1/2 of expansion

        # shift values, takes maximum value ini_shift_angles, take sqrt to
        # sample equally over full sphere
        shift = rng3.random(n_repeat)*shift_angles

        shift[0] = 0  # set first shift to zero for qfi
        for i in range(n_repeat):
            random_vector_normed[i] = (
                random_vector_normed[i]
                / np.sqrt(np.sum(np.abs(random_vector_normed[i])**2))
            )
        ini_angles = (
            unshifted_ini_angles
            + np.array([
                random_vector_normed[i]*shift[i] for i in range(n_repeat)
            ])
        )

    # unpack and reshape angles
    output = None
    for ridx in range(n_repeat):

        angle_list = []
        for j in range(depth):
            for k in range(n_qubits):
                type_pauli = ini_pauli[j][k]
                if type_pauli != 0:
                    angle_list.append(ini_angles[ridx][j][k])

        if output is None:
            output = np.zeros((n_repeat, len(angle_list)))

        output[ridx, :] = np.array(angle_list)

    return output


def _get_crossfidelity_circuits_standard(
    cross_fid, X_data, prefix, progress_bar=True, idx_offset=0,
):
    """ """
    X_data = np.atleast_2d(X_data)

    if progress_bar:
        _iterator = tqdm(
            enumerate(X_data),
            desc='making '+prefix+' circuits',
            total=len(X_data)
        )
    else:
        _iterator = enumerate(X_data)

    bound_circs = []
    for idx, data_item in _iterator:
        circs = cross_fid.bind_params_to_meas(data_item)
        for tmp in circs:
            # tmp = simplify_rotation_angles(tmp)
            tmp.name = prefix + f'{idx_offset+idx}' + '-' + tmp.name
            bound_circs.append(tmp)

    return bound_circs


def _get_crossfidelity_circuits_custom_angles(
    cross_fid, X_data, prefix, crossfid_angles, progress_bar=True,
    idx_offset=0,
):
    """ """
    X_data = np.atleast_2d(X_data)

    if progress_bar:
        _iterator = tqdm(
            enumerate(X_data),
            desc='making '+prefix+' circuits',
            total=len(X_data)
        )
    else:
        _iterator = enumerate(X_data)

    bound_circs = []
    for didx, data_item in _iterator:
        # iterate over crossfidelity angles
        for cfidx, cfangles in enumerate(crossfid_angles):

            # concat data angles and measurement angles
            _full_angles = np.concatenate(
                (
                    data_item,          # data angles
                    cfangles[:, 1],     # z-angles
                    cfangles[:, 0],     # y-angles
                )
            )

            circs = cross_fid.bind_params_to_meas(_full_angles)
            assert len(circs) == 1

            for tmp in circs:
                tmp.name = (
                    prefix + f'{idx_offset+didx}' + '-CrossFid'
                    + f'{cfidx}'
                )
                bound_circs.append(tmp)

    return bound_circs


def _make_ansatz_and_crossfid(
    n_qubits,
    depth,
    n_features,
    type_circuit,
    circuit_initial_angles,
    circuit_random_seed,
    transpile_instance,
    transpiler,
    crossfid_mode,
    n_unitaries,
    crossfid_random_seed,
    n_bootstraps,
):
    """ """

    # make PQC and pre-transpile
    pqc, pqc_params = make_natural_parameterised_circuit(
        n_qubits, depth, n_features, type_circuit=type_circuit,
        initial_angles=circuit_initial_angles, random_seed=circuit_random_seed,
    )

    if isinstance(crossfid_mode, str) and crossfid_mode == 'inverse':

        # this is a special mode where the inverse pqc at angles=0 is used as
        # the crossfid basis
        if n_unitaries != 1:
            raise ValueError(
                'In "inverse" crossfidelity mode, n_unitaries should be one')

        # edit circuit to append central inverse projection
        tmp = pqc.compose(
            pqc.inverse().bind_parameters(np.zeros(len(pqc.parameters)))
        )
        pqc = tmp

        # use internal 'identity' mode of crossfid class
        _crossfid_mode = 'identity'
        _n_unitaries = n_unitaries

    elif isinstance(crossfid_mode, np.ndarray):

        # this is a special mode where the cross-fidelity angles are specified
        # by hand
        if not np.array_equal(crossfid_mode.shape, [n_unitaries, n_qubits, 3]):
            raise ValueError(
                'crossfid_mode array has wrong shape, expected '
                + f'{(n_unitaries, n_qubits, 3)}'+' got shape: '
                + f'{crossfid_mode.shape}'
            )

        # edit circuit to append parameterised Rz and Ry layers
        for qidx in range(n_qubits):
            new_param = Parameter('R'+str(len(pqc_params)))
            pqc_params.append(new_param)
            pqc.rz(new_param, qidx)
        for qidx in range(n_qubits):
            new_param = Parameter('R'+str(len(pqc_params)))
            pqc_params.append(new_param)
            pqc.ry(new_param, qidx)

        # use internal 'identity' mode of crossfid class
        _crossfid_mode = 'identity'
        _n_unitaries = 1

    else:
        _crossfid_mode = crossfid_mode
        _n_unitaries = n_unitaries

    logger.info('transpiling PQC...')
    ansatz = _pretranspile_pqc(pqc, transpile_instance, transpiler)
    # strip optimisation level part from name
    if 'pytket' in transpiler:
        _transpiler = 'pytket'
    else:
        _transpiler = transpiler

    # make circuits
    logger.info('binding circuits...')
    cross_fid = CrossFidelity(
        ansatz=ansatz,
        instance=transpile_instance,
        nb_random=_n_unitaries,
        transpiler=_transpiler,
        seed=crossfid_random_seed,
        num_bootstraps=n_bootstraps,
        mode=_crossfid_mode,
    )

    return ansatz, cross_fid


def _execute_batch(
    measurement_error_mitigation,
    ansatz,
    circuit_buffer,
    n_unitaries,
    circuit_dump_name,
    exe_instance,
    data_points_in_buffer,
    local_args,
    results_name,
):
    """ """

    # in this mode always use tensored_meas_cal since can submit on a
    # job by job basis
    meas_err_mit_assets = None
    if measurement_error_mitigation == 1:
        meas_err_mit_assets = {}

        mit_pattern = [[x] for x in ansatz._transpiler_map.values()]
        mit_circs, _ = tensored_meas_cal(
            mit_pattern,
            qr=circuit_buffer[0].qregs[0],
            cr=circuit_buffer[0].cregs[0],
        )
        circuit_buffer = circuit_buffer + mit_circs
        meas_err_mit_assets['mit_pattern'] = mit_pattern

    # test we have the number of circuits we expect
    if not len(circuit_buffer) == len(data_points_in_buffer)*n_unitaries+2:
        raise ValueError(
            'Expected '+f'{len(data_points_in_buffer)*n_unitaries+2}'
            + ' circuits, but got '+f'{len(circuit_buffer)}'
        )

    # execute batch
    if circuit_dump_name is not None:
        _dump_output(
            circuit_dump_name+'/batched_data'+f'{data_points_in_buffer}',
            circuit_buffer
        )
    results = exe_instance.execute(circuit_buffer, had_transpiled=True)

    # iterate over results, splitting and outputting
    for didx in data_points_in_buffer:

        tmp_results = copy.deepcopy(results)
        tmp_results.results = [
            res for res in results.results
            if 'data'+f'{didx}' in res.header.name
            or 'cal_' in res.header.name
        ]

        # joblib results dump
        logger.info('saving results...')
        _dump_ibmq_crossfid_results(
            tmp_results.to_dict(),
            ansatz._transpiler_map,
            ansatz._transpiled_circuit,
            meas_err_mit_assets,
            local_args,
            results_name+'/data'+f'{didx}',
        )


def get_ibmq_crossfid_results(
    backend_name,
    n_qubits,
    depth,
    type_circuit,
    type_dataset,
    n_shots,
    n_unitaries,
    n_repeat=None,
    rescale_factor=1,
    n_pca_features=0,
    crossfid_mode='1qHaar',
    n_bootstraps=0,
    random_seed=1,
    circuit_initial_angles='natural',
    circuit_random_seed=None,
    data_random_seed=None,
    crossfid_random_seed=None,
    results_name='results',
    apply_stratify=True,
    transpiler='pytket',
    hub='ibm-q',
    group='open',
    project='main',
    measurement_error_mitigation=1,
    backend_options=None,
    initial_layout=None,
    simulate_ibmq=0,
    noise_model=None,
    seed_simulator=None,
    data_vars_dump_name=None,
    circuit_dump_name=None,
    data_batch_size=1,
    data_slice_start=None,
    data_slice_end=None,
):
    """
    Parameters
    ----------
    backend_name : str
        Name of backend to execute on
    n_qubits : int
        Number of qubits to use in the PQC circuit
    depth : int
        Depth of the PQC circuit
    type_circuit : int
        Options:
            0: natural parameterized quantum circuit (NPQC)
            1: NPQC without ring
            2: NPQC ring with additional SWAP and 4 parameters (special case)
            3: YZ CNOT alternating circuit
    type_dataset : int
        Dataset to use, options:
            0: breast cancer
            1: make_classification dataset
            2: circles dataset
            3: handwriting two digits
            4: handwriting all digits
            5: random data
    n_shots : int
        Number of measurment shots for fidelity estimations
    n_unitaries : int
        Number of unitaries for crossfidelity estimation
    n_repeat : int, optional
        Ignored unless type_dataset==5, in which case it sets the number of
        random points to generate
    rescale_factor : float, optional
        Additional rescale of variables, equivalent to width of Gaussian,
        large: underfitting, small: overfitting
    n_pca_features : int, optional
        If set to a number > 0, the data will be preproccesed using PCA with
        that number of principal components
    crossfid_mode : str OR numpy.ndarray, optional

        How to generate the random measurements, supported str options:
            'identity' : trivial case, do nothing
            '1qHaar' : single qubit Haar random unitaries, generated using
                       qiskit's random unitary function
            'rypiOver3' : 1/3 of qubits are acted on by identities, 1/3 by
                          Ry(pi/3), and 1/3 by Ry(2pi/3)
            'inverse' : special case for natural pqc's, use the "central"
                        (angles=0) state as the single measurement basis
            'RzRy' : single qubit Haar random unitaries, generated from
                     selecting euler angles using numpy random functions
                     instead of qiskit random unitary function

        If a numpy array is passed this will be used to generate the RzRy
        measurement angles. The array must have shape:
            (n_unitaries, n_qubits, 3)
        where [:,:,0] contains the Rz angles, and [:,:,1] the Ry angles.

    n_bootstraps : int, optional
        Number of bootstrap resamples to use to estimate error on CrossFidelity
    random_seed : int, optional
        Random seed for reproducibility
    circuit_initial_angles : {'natural', 'random', 'zeros'}, optional
        Angles to centre feature parameters around, passed to PQC construction
    circuit_random_seed : int or None
        Random seed for reproducibility, passed to PQC construction function.
        If set to None defaults to the value of `random_seed`
    data_random_seed : int or None
        Random seed for reproducibility, passed to scikit-learn functions. If
        set to None defaults to the value of `random_seed`
    crossfid_random_seed : int or None
        Random seed for reproducibility, passed to crossfidelity obj. If set to
        None defaults to the value of `random_seed`
    results_name : str
        Filename for results dump
    apply_stratify : boolean, optional
        If True, test/train split is stratified
    transpiler : str, optional
            Choose how to transpile circuits, current options are:
                'instance' : use quantum instance
                'pytket' : use pytket compiler at optimisation level 2
                'pytket_2' : use pytket compiler at optimisation level 2
                'pytket_1' : use pytket compiler at optimisation level 1
                'pytket_0' : use pytket compiler at optimisation level 0
    hub : str
        (Qiskit) User's IBMQ access information, defaults to public access
    group : str
        (Qiskit) User's IBMQ access information, defaults to public access
    project : str
        (Qiskit) User's IBMQ access information, defaults to public access
    measurement_error_mitigation : int, optional
        (Qiskit) Flag for whether or not to use measurement error mitigation.
    backend_options : dict, or None
        (Qiskit) Passed to QuantumInstance
    initial_layout : list, or None
        (Qiskit) Passed to QuantumInstance
    simulate_ibmq : int, default 0
        Exposes the arg of make_quantum_instance, allowing noisy simulation
    noise_model : noise model, or None
        (Qiskit) Passed to QuantumInstance
    seed_simulator : int, or None
        (Qiskit) Passed to QuantumInstance
    data_vars_dump_name : str, optional
        If not set to None, data variables will be dumped as joblib here
    circuit_dump_name : str, optional
        If not set to None, executed circuits will be dumped as joblib here
    data_batch_size : int, optional
        If set, this number of data points will be batched together for
        execution
    data_slice_start : int, optional
        If not None, the full dataset will be sliced using this lower bound
        with python list slicing convention
        i.e. data -> data[data_slice_start:]
    data_slice_end : int, optional
        If not None, the full dataset will be sliced using this upper bound
        with python list slicing convention
        i.e. data -> data[:data_slice_end]
    """
    all_args = copy.copy(locals())

    # use value of shared random seed if separate seeds not passed
    if circuit_random_seed is None:
        circuit_random_seed = random_seed
    if data_random_seed is None:
        data_random_seed = random_seed
    if crossfid_random_seed is None:
        crossfid_random_seed = random_seed

    # load and preprocess data
    if type_dataset in range(5):

        X_data, y_data, X_test, y_test, X_all, y_all = _get_data_vars(
            type_dataset,
            None,
            None,
            n_pca_features,
            rescale_factor,
            data_random_seed,
            apply_stratify,
        )
        if data_vars_dump_name is not None:
            _dump_output(
                data_vars_dump_name,
                {
                    'X_train': X_data,
                    'y_train': y_data,
                    'X_test': X_test,
                    'y_test': y_test,
                    'X_all': X_all,
                    'y_all': y_all,
                }
            )

    elif type_dataset == 5:

        # special case, for random angle data
        if circuit_initial_angles != 'zeros':
            raise ValueError(
                'For random data, circuit_initial_angles must be set'
                + ' to "zeros".'
            )
        X_data = _make_random_angles(type_circuit, n_repeat, depth, n_qubits)
        if data_vars_dump_name is not None:
            _dump_output(
                data_vars_dump_name,
                {
                    'X_train': None,
                    'y_train': None,
                    'X_test': None,
                    'y_test': None,
                    'X_all': X_data,
                    'y_all': np.array([0 for _ in range(len(X_data))]),
                }
            )

    else:
        raise ValueError('type_dataset not recognised: '+f'{type_dataset}')
    n_data, n_features = X_data.shape

    if data_slice_start is None:
        data_slice_start = -1
    if data_slice_end is None:
        data_slice_end = n_data+1

    # make instances
    exe_instance, transpile_instance = _get_quantum_instances(
        backend_name, n_shots, backend_options, initial_layout,
        bool(simulate_ibmq), noise_model, seed_simulator, hub, group, project,)

    ansatz, cross_fid = _make_ansatz_and_crossfid(
        n_qubits,
        depth,
        n_features,
        type_circuit,
        circuit_initial_angles,
        circuit_random_seed,
        transpile_instance,
        transpiler,
        crossfid_mode,
        n_unitaries,
        crossfid_random_seed,
        n_bootstraps,
    )

    # iterate over data items
    logger.info('running on device...')
    circuit_buffer = []
    data_points_in_buffer = []
    for dataidx, X_val in enumerate(tqdm(X_data)):
        if (
            dataidx >= data_slice_start
            and dataidx < data_slice_end
        ):

            local_args = copy.copy(all_args)
            local_args['X_val'] = X_val

            if isinstance(crossfid_mode, str):
                _circuits = _get_crossfidelity_circuits_standard(
                    cross_fid, X_val, 'data', progress_bar=False,
                    idx_offset=dataidx,
                )
            elif isinstance(crossfid_mode, np.ndarray):
                _circuits = _get_crossfidelity_circuits_custom_angles(
                    cross_fid, X_val, 'data', crossfid_mode,
                    progress_bar=False, idx_offset=dataidx,
                )
            else:
                raise TypeError(
                    'unrecognized type for crossfid_mode: '
                    + f'{type(crossfid_mode)}'
                )

            if circuit_dump_name is not None:
                _dump_output(circuit_dump_name+'/data'+f'{dataidx}', _circuits)

            circuit_buffer = circuit_buffer + _circuits
            data_points_in_buffer.append(dataidx)

            # if collected a full batch, execute
            if len(data_points_in_buffer) == data_batch_size:
                _execute_batch(
                    measurement_error_mitigation,
                    ansatz,
                    circuit_buffer,
                    n_unitaries,
                    circuit_dump_name,
                    exe_instance,
                    data_points_in_buffer,
                    local_args,
                    results_name,
                )
                circuit_buffer = []
                data_points_in_buffer = []
            elif len(data_points_in_buffer) > data_batch_size:
                raise ValueError(
                    'something has gone wrong! data batch is oversized.'
                )

    # run final incomplete batch
    if len(data_points_in_buffer) > 0:
        _execute_batch(
            measurement_error_mitigation,
            ansatz,
            circuit_buffer,
            n_unitaries,
            circuit_dump_name,
            exe_instance,
            data_points_in_buffer,
            local_args,
            results_name,
        )
