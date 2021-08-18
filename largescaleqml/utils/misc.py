"""
"""

import os
import copy
import joblib
import logging
from tqdm import tqdm

from qiskit.result import Result
from qiskit.ignis.mitigation import TensoredMeasFitter

from qcoptim.utilities import FastCountsResult

logger = logging.getLogger(__name__)


def _dump_output(name, output):
    """ """

    # make directory if it doesn't exist
    filename = name+'.joblib'
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # pickle output obj using joblib
    joblib.dump(output, filename, compress=True)


#
# used in applying measurement error mitigation scripts under studies
#


def _apply_measurement_error_mitigation(
    job_filename,
    raw_directory,
    mem_directory,
):
    """ """
    # get list of raw files to process
    files = _find_results_files(
        job_filename, raw_directory, mem_directory,
    )
    # (check all unique)
    assert len(files) == len(set(files))

    if len(files) > 0:
        print('found '+f'{len(files)}'+' files, e.g. '+f'{files[0]}')

        # apply measurement error mitigation
        for file in tqdm(files, desc='applying m.e.m.'):
            _process_file(
                file,
                job_filename, raw_directory, mem_directory,
            )

    else:
        print('found '+f'{len(files)}'+' files')

    # copy X and y variables file
    X_y_vars = joblib.load(
        '/'.join([raw_directory, job_filename, 'X_y_vars.joblib'])
    )
    joblib.dump(
        X_y_vars,
        '/'.join([mem_directory, job_filename, 'X_y_vars.joblib']),
        compress=True,
    )
    


def _find_results_files(
    search_sub_directory,
    raw_directory,
    mem_directory,
    check_for_output=True
):
    """ """
    files = []

    for file in os.scandir(
        os.path.join(raw_directory, search_sub_directory)
    ):
        fullpath = file.path
        filename = os.path.basename(fullpath)
        split_filename = os.path.splitext(filename)[0]

        # check that is IBMQ data
        if 'data' in split_filename:
            loaded_file = joblib.load(fullpath)

            # check that contains measurement error mitigation data
            if 'meas_err_mit_assets' in loaded_file.keys():

                # check file doesn't already exist in meas_err_mit dir
                if check_for_output and not os.path.isfile(os.path.join(
                    mem_directory, search_sub_directory, filename
                )):
                    files.append(filename)

    return files


def _process_file(
    filename,
    search_sub_directory,
    raw_directory,
    mem_directory,
):
    """ """
    input_fullpath = os.path.join(
        raw_directory, search_sub_directory, filename)
    output_fullpath = os.path.join(
        mem_directory, search_sub_directory, filename)

    # load data and train fitter
    loaded_data = joblib.load(input_fullpath)
    loaded_results = Result.from_dict(loaded_data['qiskit-results'])
    fitter = TensoredMeasFitter(
        loaded_results,
        **loaded_data['meas_err_mit_assets']
    )
    # assert max([fitter.readout_fidelity(idx) for idx in range(8)]) < 1.

    # crop results to just data results
    _results = [
        res for res in loaded_results.results
        if 'data' in res.header.name
    ]
    loaded_results.results = _results

    # apply filter
    filtered_results = fitter.filter.apply(loaded_results)

    # make output
    output_data = copy.deepcopy(loaded_data)
    output_data['qiskit-results'] = filtered_results

    # make directory if needed
    directory = os.path.dirname(output_fullpath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    joblib.dump(output_data, output_fullpath, compress=True)


#
# used in process results scripts under studies
#


def _load_data(
    results_sub_directory,
    start_idx,
    end_idx,
):
    """ """
    loaded_results = []
    for dataidx in tqdm(
        range(start_idx, end_idx),
        desc='loading '+f'{results_sub_directory}'+' data'
    ):
        data = joblib.load(
            results_sub_directory+'/data'+f'{dataidx}'+'.joblib')
        
        # cast results from dict if needed
        tmp = data['qiskit-results']
        if isinstance(tmp, dict):
            tmp = Result.from_dict(tmp)
        elif not isinstance(tmp, Result):
            raise TypeError('type of results not recognised: '+f'{type(tmp)}')
        
        loaded_results.append(FastCountsResult(tmp))

    return loaded_results
