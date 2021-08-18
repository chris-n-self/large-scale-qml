"""
Process a set of measurment results into the cross-fidelity estimate of the
Gram matrix for the data
"""

import os
import joblib

import numpy as np

from largescaleqml import _process_crossfid_results
from largescaleqml.utils import _load_data

# load all job arguments from job_specification file
from JOB_SPECIFICATION import (
    N_BOOTSTRAPS,
    N_UNITARIES,
    JOB_FILENAME,
    N_REPEAT,
)


if __name__ == '__main__':

    # number of data items
    N_DATA = N_REPEAT

    SOURCE = 'raw'  # no measurement error mitigation
    # SOURCE = 'meas_err_mit'  # measurement error mitigation

    LOAD_FILENAME = (
        '/'.join([
            'results',
            'unprocessed',
            SOURCE,
            JOB_FILENAME,
        ])
    )

    # load data
    loaded_results = []
    loaded_results += _load_data(
        LOAD_FILENAME, 0, N_DATA,
    )

    # process cross-fidelity
    crossfid_gram, purities = _process_crossfid_results(
        loaded_results, N_UNITARIES, N_DATA, N_BOOTSTRAPS, 0,
        prefix='data',
    )

    # save output
    _directory = '/'.join([
        'results',
        'processed',
        SOURCE,
    ])
    if not os.path.exists(_directory):
        os.makedirs(_directory)
    np.savetxt(
        '/'.join([
            _directory,
            'GramMatrix'+'-'+JOB_FILENAME+'.csv',
        ]),
        crossfid_gram, delimiter=',',
    )

    # expose X and y data vectors for simplicity
    X_y_vars = joblib.load(LOAD_FILENAME+'/X_y_vars.joblib')
    np.savetxt(
        '/'.join([
            _directory,
            'X'+'-'+JOB_FILENAME+'.csv',
        ]),
        X_y_vars['X_all'][:N_DATA], delimiter=',',
    )
    np.savetxt(
        '/'.join([
            _directory,
            'y'+'-'+JOB_FILENAME+'.csv',
        ]),
        X_y_vars['y_all'][:N_DATA], delimiter=',',
    )
