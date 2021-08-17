"""
"""

#
# Set all run arguments
#

BACKEND_NAME = 'aer_simulator'
N_QUBITS = 8
DEPTH = 10
TYPE_CIRCUIT = 3
TYPE_DATASET = 5
N_REPEAT = 100
APPLY_STRATIFY = True
RESCALE_FACTOR = 1.
N_PCA_FEATURES = 0
N_BOOTSTRAPS = 0
CIRCUIT_RANDOM_SEED = None
DATA_RANDOM_SEED = None
CROSSFID_RANDOM_SEED = 53
CIRCUIT_INITIAL_ANGLES = 'zeros'
CROSSFID_MODE = 'RzRy'
N_SHOTS = 8192
N_UNITARIES = 50
DATA_BATCH_SIZE = 10

# user's IBMQ access
HUB = 'ibm-q'
GROUP = 'open'
PROJECT = 'main'

# used to name log files
JOB_TAG = 'randomdata_YZ'

# make output filename
JOB_FILENAME = (
    ','.join([
        BACKEND_NAME,
        'n_qubits'+f'{N_QUBITS}',
        'depth'+f'{DEPTH}',
        'n_shots'+f'{N_SHOTS}',
        'n_unitaries'+f'{N_UNITARIES}',
        'crossfid_mode'+f'{CROSSFID_MODE}',
        'n_repeat'+f'{N_REPEAT}',
    ])
)
