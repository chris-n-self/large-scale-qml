
Code to accompany "Large-scale quantum machine learning" (https://arxiv.org/abs/2108.01039). 

## Setup

Run these commands in the base directory. 

Create a new conda environment called `large-scale-qml` and install all the packages needed:

```sh
$ conda env create -f large-scale-qml.yml
$ conda activate large-scale-qml
$ pip install -r requirements.txt
```

Then, install the local package, called `largescaleqml`:

```sh
$ pip install -e . 
```

## Running case studies

In the 'studies' directory are the two examples used in the paper: a random dataset (Fig.2) and scikit-learn's handwritten digit classification (Fig.4). These have 'NPQC' and 'YZ' subdirectories for the different parameterised circuit types.

Execution is broken into several steps -- first collecting the measurment results from an IBMQ backend or simulator, then (optionally) applying measurement error mitigation to these results, next processing the measurement results to obtain the Gram matrix, and finally fitting the support vector machine classifier using the Gram matrix. These different tasks are carried out by different scripts and all of the intermediate data is saved to disk. 

Each job folder, e.g. 'handwriting-all-digits/NPQC', contains the scripts: 'collect-results.py', 'apply_meas_error_mit.py' and 'process-results.py' (explained below) as well as a file 'JOB_SPECIFICATION.py'. The job specification file sets the values of variables used in an execution run, each of the other scripts will import the variables they need from that file. The variables defined in 'JOB_SPECIFICATION.py' mostly relate to the function `largescaleqml.calculations_qpu.get_ibmq_crossfid_results` and their meaning is explained in the docstring of that function (reproduced at the bottom of this page).

In addition to setting the execution variables, we also define how output files are named in 'JOB_SPECIFICATION.py'. For example,
```python
JOB_FILENAME = (
    ','.join([
        BACKEND_NAME,
        'n_qubits'+f'{N_QUBITS}',
        'depth'+f'{DEPTH}',
        'n_shots'+f'{N_SHOTS}',
        'n_unitaries'+f'{N_UNITARIES}',
        'crossfid_mode'+f'{CROSSFID_MODE}',
    ])
)
```
combines several of the execution variables into a readable output name. If we wanted to additionally experiment with varying other variables, e.g. `CROSSFID_RANDOM_SEED` we could add that variable into this output name.

The overall idea is that all arguments are fixed in 'JOB_SPECIFICATION.py', each script is executed in sequence to produce the final output, then 'JOB_SPECIFICATION.py' is changed and the whole process is repeated.

### 1. Collecting measurement results `collect-results.py`

This script executes the encoding circuits and carries out the randomised measurements for each data item on the qiskit backend. 

The qiskit measurement results are saved in a subdirectory 'results/unprocessed/raw' (which will be created inside the directory if it does not exist). Inside 'raw' a folder will be created for the run using `JOB_FILENAME` imported from 'JOB_SPECIFICATION.py'. The measurement results of each data point will be pickled and saved in a separate file, e.g. 'data0.joblib'.

This step will also generate a number of temporary files inside the 'tmp' directory including pickled copies of the circuits that were executed.

### 2. (Optional) Applying measurement error mitigation `apply_meas_error_mit.py`

This script can be run to apply measurement error mitigation to a previously collected set of results. The script will look for the un-mitigated results in the folder 'results/unprocessed/raw/JOB_FILENAME', where `JOB_FILENAME` is imported from 'JOB_SPECIFICATION.py'. When mitigation is applied a copy of the `JOB_FILENAME` folder is created in 'results/unprocessed/meas_err_mit' with the mitigated results files.

The measurement error mitigation applied is qiskit's [tensored mitigation](https://qiskit.org/documentation/stubs/qiskit.ignis.mitigation.TensoredMeasFitter.html) using a single qubit tensored noise model. This type of measurement error mitigation is very cheap and the calibration circuits are remeasured with every circuit execution batch.

### 3. Process results into Gram matrix `process-results.py`

This script processes the randomised measurement results to compute the Gram matrix. This is the unmitigated Gram matrix, Eqn(6) of the paper not Eqn(7). 

This script has two additional variables `N_DATA` that sets the number of data items, and `SOURCE` that toggles between no measurement error mitigation `SOURCE='raw'` and with measurement error mitigation `SOURCE = 'meas_err_mit'`.

## Additional notes

### Qiskit circuit execution batching

Circuits are broken into batches for submission to IBM Quantum. The different measurement circuits for each data point are never separated, so the unit of batching is data points. The variable `DATA_BATCH_SIZE` in 'JOB_SPECIFICATION.py' controls how many data points are included into each batch. 

It is recommended to ensure that the batch size is small enough that a single batch contains less circuits than the backend's maximum circuit count. If this is not the case everything will still work correctly (since qiskit's QuantumInstance class is being used as an internal executor), however measurement error mitigation calibration circuits are only inserted once into each batch.

### Logging and temporary files

The 'collect-results.py' script generates log files using python's logging package at debug level. Both qiskit and the `largescaleqml` package in this repository print logging information to this file.

Additionally, 'collect-results.py' generates a number of temporary files that can be inspected afterwards. These go into e.g. 'handwriting-all-digits/NPQC/tmp' and are named using `JOB_FILENAME`, imported from 'JOB_SPECIFICATION.py', and the date & time. These include the exact circuits that were executed on the backend ('circuits' subdirectory), the data variables ('data_vars') and the logical to physical transpiler mapping ('transpiles').

### `largescaleqml.calculations_qpu.get_ibmq_crossfid_results` docstring

The function `largescaleqml.calculations_qpu.get_ibmq_crossfid_results` explains the meaning of most of the variables set in 'JOB_SPECIFICATION.py'. Its docstring and default arguments are reproduced here:

```python
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
```
