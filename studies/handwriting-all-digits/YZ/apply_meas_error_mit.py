"""
Apply measurement error mitigation to a collected set of results
"""

import os
import copy
import joblib
from tqdm import tqdm

from largescaleqml.utils import _find_results_files, _process_file

from JOB_SPECIFICATION import JOB_FILENAME


if __name__ == '__main__':

    # get list of raw files to process
    files = _find_results_files(
        JOB_FILENAME,
        'results/unprocessed/raw',
        'results/unprocessed/meas_err_mit',
    )
    # (check all unique)
    assert len(files) == len(set(files))

    if len(files) > 0:
        print('found '+f'{len(files)}'+' files, e.g. '+f'{files[0]}')

        # apply measurement error mitigation
        for file in tqdm(files, desc='applying m.e.m.'):
            _process_file(
                file,
                JOB_FILENAME,
                'results/unprocessed/raw',
                'results/unprocessed/meas_err_mit',
            )

    else:
        print('found '+f'{len(files)}'+' files')
