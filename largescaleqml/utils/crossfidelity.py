"""
"""

import joblib
import logging
from tqdm import tqdm
from itertools import product
from tempfile import TemporaryDirectory

import numpy as np

from qiskit.result import Result

from qcoptim.cost.crossfidelity import (
    _crosscorrelation_per_u,
    _purity_per_u,
)
from qcoptim.utilities import (
    resample_histogram,
    bootstrap_resample,
    FastCountsResult,
)


logger = logging.getLogger(__name__)


def _compute_purity(
    results, prefix, n_unitaries, n_bootstraps, random_seed, counts_transform,
    vectorise=True, force_fast_counts=True, 
):
    """ """

    if isinstance(results, Result):
        tmp_results = results
    else:
        raise TypeError(
            'results type not recognised: '+f'{type(results)}')

    if force_fast_counts and not isinstance(results, FastCountsResult):
        tmp_results = FastCountsResult(results)

    dist_tr_rho_2 = _purity_per_u(
        tmp_results,
        n_unitaries,
        names=(lambda idx: prefix + f'{idx}'),
        vectorise=vectorise,
        counts_transform=counts_transform,
    )

    if n_bootstraps > 0:
        tr_rho_2, _ = bootstrap_resample(
            np.mean, dist_tr_rho_2, n_bootstraps,
            random_seed=random_seed,
        )
    else:
        tr_rho_2 = np.mean(dist_tr_rho_2)

    return tr_rho_2


def _compute_crosscorrelation(
    resultsA, prefixA, resultsB, prefixB,
    n_unitaries, n_bootstraps, random_seed,
    counts_transform, vectorise=True, force_fast_counts=True,
):
    """ """

    if isinstance(resultsA, Result):
        tmp_resultsA = resultsA
    else:
        raise TypeError(
            'results type not recognised: '+f'{type(resultsA)}')
    if isinstance(resultsB, Result):
        tmp_resultsB = resultsB
    else:
        raise TypeError(
            'results type not recognised: '+f'{type(resultsB)}')

    if force_fast_counts and not isinstance(resultsA, FastCountsResult):
        tmp_resultsA = FastCountsResult(resultsA)
    if force_fast_counts and not isinstance(resultsB, FastCountsResult):
        tmp_resultsB = FastCountsResult(resultsB)

    dist_tr_rhoA_rhoB = _crosscorrelation_per_u(
        tmp_resultsA, tmp_resultsB, n_unitaries,
        circ_namesA=lambda idx: prefixA + f'{idx}',
        circ_namesB=lambda idx: prefixB + f'{idx}',
        vectorise=vectorise,
        counts_transform=counts_transform,
    )

    if n_bootstraps > 0:
        tr_rhoA_rhoB, _ = bootstrap_resample(
            np.mean, dist_tr_rhoA_rhoB, n_bootstraps,
            random_seed=random_seed,
        )
    else:
        tr_rhoA_rhoB = np.mean(dist_tr_rhoA_rhoB)

    return tr_rhoA_rhoB


def _factory_retrieve_results(results):
    """ """
    if isinstance(results, (Result, FastCountsResult)):
        logger.debug('\t'+'\t'+'(parsing results as Result)')

        def _retrieve_result(idx):
            return results

    elif (
        isinstance(results, list) 
        and isinstance(results[0], (Result, FastCountsResult))
    ):
        logger.debug('\t'+'\t'+'(parsing results as list of Results)')

        def _retrieve_result(idx):
            return results[idx]

    elif (
        isinstance(results, list)
        and isinstance(results[0], TemporaryDirectory)
    ):
        logger.debug(
            '\t'+'\t'+'(parsing results as list of TemporaryDirectory)'
        )

        def _retrieve_result(idx):
            return joblib.load(results[idx].name+'/batch0.joblib')

    else:
        if isinstance(results, list):
            raise TypeError(
                'results type not recognised, recieved list of '
                + f'{type(results[0])}'
            )
        else:
            raise TypeError(
                'results type not recognised, recieved ' + f'{type(results)}'
            )

    return _retrieve_result


def _process_crossfid_results(
    results,
    n_unitaries,
    n_training,
    n_bootstraps,
    random_seed,
    downsample_shots_to=None,
    prefix='data',
    vectorise=True,
    force_fast_counts=True,
):
    """ """

    _retrieve_result = _factory_retrieve_results(results)

    # optionally downsample shots
    if downsample_shots_to is not None:
        def counts_transform(counts_dict):
            return dict(zip(counts_dict.keys(), resample_histogram(
                np.array(list(counts_dict.values())),
                downsample_shots_to,
                random_seed=random_seed,
            )))
    else:
        counts_transform = None

    # construct kernel matrix from fidelities
    crossfid_train_gram = np.zeros([n_training, n_training])
    purities = np.zeros(n_training)
    logger.debug('\t'+'processing crossfidelity training results...')
    for i, j in tqdm(
        product(range(n_training), range(n_training)),
        total=n_training*n_training,
        desc='proc. crossfid.'
    ):
        if j < i:
            continue
        if j == i:
            # purities[i] = _compute_purity(
            #     _retrieve_result(i), prefix + f'{i}' + '-CrossFid',
            #     n_unitaries, n_bootstraps, random_seed, counts_transform,
            #     vectorise=vectorise, force_fast_counts=force_fast_counts,
            # )
            purities[i] = _compute_crosscorrelation(
                _retrieve_result(i), prefix+f'{i}'+'-CrossFid',
                _retrieve_result(i), prefix+f'{i}'+'-CrossFid',
                n_unitaries, n_bootstraps, random_seed, counts_transform,
                vectorise=vectorise, force_fast_counts=force_fast_counts,
            )
        else:
            crossfid_train_gram[i, j] = _compute_crosscorrelation(
                _retrieve_result(i), prefix+f'{i}'+'-CrossFid',
                _retrieve_result(j), prefix+f'{j}'+'-CrossFid',
                n_unitaries, n_bootstraps, random_seed, counts_transform,
                vectorise=vectorise, force_fast_counts=force_fast_counts,
            )

    # normalise with max purity
    # for i, j in product(range(n_training), range(n_training)):
    #     if j <= i:
    #         continue
    #     crossfid_train_gram[i, j] = (
    #         crossfid_train_gram[i, j] / max(purities[i], purities[j])
    #     )

    # symmetrise
    crossfid_train_gram = (
        crossfid_train_gram + np.transpose(crossfid_train_gram)
    )
    # diagonal is purities values
    crossfid_train_gram = (
        crossfid_train_gram + np.diag(purities)
    )

    return crossfid_train_gram, purities
