"""
"""

#
# Set all run arguments
#

BACKEND_NAME = 'aer_simulator'
N_QUBITS = 8
DEPTH = 10
TYPE_CIRCUIT = 3
TYPE_DATASET = 4
APPLY_STRATIFY = True
RESCALE_FACTOR = 1.
N_PCA_FEATURES = 0
N_BOOTSTRAPS = 0
CIRCUIT_RANDOM_SEED = 31
DATA_RANDOM_SEED = 0
CROSSFID_RANDOM_SEED = 53
CIRCUIT_INITIAL_ANGLES = 'random'
CROSSFID_MODE = 'RzRy'
N_SHOTS = 8192
N_UNITARIES = 8
DATA_BATCH_SIZE = 10

# user's IBMQ access
HUB = 'ibm-q'
GROUP = 'open'
PROJECT = 'main'

# used to name log files
JOB_TAG = 'allhandwriting_YZ'

# make output filename
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
