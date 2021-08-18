"""
Apply measurement error mitigation to a collected set of results
"""

from largescaleqml.utils import _apply_measurement_error_mitigation

# load file location from job specification
from JOB_SPECIFICATION import JOB_FILENAME


if __name__ == '__main__':

    _apply_measurement_error_mitigation(
        JOB_FILENAME,
        'results/unprocessed/raw',
        'results/unprocessed/meas_err_mit',
    )
