"""
"""

import logging
from datetime import datetime

from largescaleqml import get_ibmq_crossfid_results

# load all job arguments from job_specification file
from JOB_SPECIFICATION import (
    BACKEND_NAME,
    JOB_TAG,
    N_QUBITS,
    DEPTH,
    TYPE_CIRCUIT,
    TYPE_DATASET,
    APPLY_STRATIFY,
    RESCALE_FACTOR,
    N_PCA_FEATURES,
    N_BOOTSTRAPS,
    CIRCUIT_RANDOM_SEED,
    DATA_RANDOM_SEED,
    CROSSFID_RANDOM_SEED,
    CIRCUIT_INITIAL_ANGLES,
    N_SHOTS,
    N_UNITARIES,
    DATA_BATCH_SIZE,
    HUB,
    GROUP,
    PROJECT,
    JOB_FILENAME,
    CROSSFID_MODE,
)

# set logging to debug
_time_now_str = datetime.now().strftime("%y%m%d-%H%M")
JOB_TRACKING_NAME = (
    BACKEND_NAME+'-'+JOB_TAG+'-'+_time_now_str
)
logging.basicConfig(filename=JOB_TRACKING_NAME+'.log', level=logging.DEBUG,)


if __name__ == '__main__':

    # set these to select slice of dataset
    DATA_SLICE_START = None
    DATA_SLICE_END = None

    _msg = (
        'backend_name='+BACKEND_NAME
    )
    print('\n'+_msg+'\n'+'-'*(len(_msg)+2))

    logging.info('='*40)
    logging.info(_msg)
    logging.info('='*40)

    FILENAME = (
        '/'.join([
            'results',
            'unprocessed',
            'raw',
            JOB_FILENAME,
        ])
    )

    # run randomised measurements on full dataset
    get_ibmq_crossfid_results(
        backend_name=BACKEND_NAME,
        n_qubits=N_QUBITS,
        depth=DEPTH,
        type_circuit=TYPE_CIRCUIT,
        type_dataset=TYPE_DATASET,
        n_shots=N_SHOTS,
        n_unitaries=N_UNITARIES,
        rescale_factor=RESCALE_FACTOR,
        n_pca_features=N_PCA_FEATURES,
        crossfid_mode=CROSSFID_MODE,
        n_bootstraps=N_BOOTSTRAPS,
        random_seed=0,
        circuit_initial_angles=CIRCUIT_INITIAL_ANGLES,
        circuit_random_seed=CIRCUIT_RANDOM_SEED,
        data_random_seed=DATA_RANDOM_SEED,
        crossfid_random_seed=CROSSFID_RANDOM_SEED,
        results_name=FILENAME,
        apply_stratify=APPLY_STRATIFY,
        transpiler='pytket_0',
        hub=HUB,
        group=GROUP,
        project=PROJECT,
        measurement_error_mitigation=1,
        backend_options=None,
        initial_layout=None,
        simulate_ibmq=0,
        noise_model=None,
        seed_simulator=None,
        data_vars_dump_name=FILENAME+'/'+'X_y_vars',
        circuit_dump_name='tmp/circuits/'+JOB_FILENAME+'-'+_time_now_str,
        data_batch_size=DATA_BATCH_SIZE,
        data_slice_start=DATA_SLICE_START,
        data_slice_end=DATA_SLICE_END,
    )
