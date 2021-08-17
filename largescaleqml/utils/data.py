"""
"""

import numpy as np

from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    make_classification,
    make_circles,
    load_breast_cancer,
    load_digits,
)
from sklearn.metrics import accuracy_score, roc_curve, auc


#
# Data set functions
# -------------------
#


def _load_dataset(type_dataset, total_dataset_size, random_seed, **kwargs):
    """ """

    if type_dataset == 0:

        # breast canter dataset
        data_cancer = load_breast_cancer()
        X_data = data_cancer["data"]
        y_data = data_cancer["target"]

    elif type_dataset == 1:

        # artficial dataset
        X_data, y_data = make_classification(
            n_samples=total_dataset_size, n_features=20,
            random_state=random_seed)

    elif type_dataset == 2:

        # ring dataset
        X_data, y_data = make_circles(
            n_samples=total_dataset_size, factor=.3, noise=.05,
            random_state=random_seed)

    elif type_dataset == 3:

        if 'select_digits' not in kwargs:
            raise ValueError(
                'Please specify digits to use, with `select_digits` arg'
            )
        select_digits = kwargs['select_digits']

        # mnist 8x8
        data_digits = load_digits()
        X_data = data_digits["data"]
        y_data = data_digits["target"]

        first_digit = (y_data == select_digits[0])
        second_digit = (y_data == select_digits[1])
        select_index = np.logical_or(first_digit,  second_digit)
        X_data = X_data[select_index]
        y_data = y_data[select_index]

        get_index_first = (y_data == select_digits[0])
        get_index_second = (y_data == select_digits[1])

        # relabel index to either 0 or 1
        y_data[get_index_first] = 0
        y_data[get_index_second] = 1
        # print(
        #     "Got in total", len(np.nonzero(select_index)[0]), "digits. For",
        #     select_digits[0], "have", len(np.nonzero(first_digit)[0]),
        #     ", for", select_digits[1], "have", 
        #     len(np.nonzero(second_digit)[0])
        # )

    elif type_dataset == 4:

        # mnist all digits
        data_digits = load_digits()
        X_data = data_digits["data"]
        y_data = data_digits["target"]

    return X_data, y_data


def _apply_pca(X_data, n_pca_features, random_seed):
    """ """

    scaler = preprocessing.StandardScaler().fit(X_data)
    X_input_data = scaler.transform(X_data)
    # pca = PCA(n_components=initial_features).fit(X_input_data)
    pca = PCA(
        n_components=n_pca_features,
        random_state=random_seed,
    ).fit(X_input_data)

    X_input_data = pca.transform(X_input_data)

    return X_input_data


def _get_data_vars(
    type_dataset,
    total_dataset_size,
    test_size_frac,
    n_pca_features,
    rescale_factor,
    random_seed,
    apply_stratify,
    **kwargs,
):
    """
    Parameters
    ----------
    type_dataset : int
        Choose numbered data set, options:
            1 : scikit-learn breast canter dataset
            2 : scikit-learn artficial classification dataset
            3 : scikit-learn handwriting, 2 digits
            4 : scikit-learn handwriting, all digits
    """

    X_input_data, y_input_data = _load_dataset(
        type_dataset, total_dataset_size, random_seed, **kwargs)
    n_features = np.shape(X_input_data)[1]

    # possibly apply PCA feature selection
    if n_pca_features > 0:
        X_input_data = _apply_pca(X_input_data, n_pca_features, random_seed)
        assert np.shape(X_input_data)[1] == n_pca_features  # sanity check
        n_features = n_pca_features

    # we have to rescale dataset in order to fit within Gaussian kernel with
    # fixed with we rescale by number of features, as well as with the
    # hyperparameter rescale_factor the width of the Gaussian is going as
    # \sqrt{n_features}
    rescale = n_features*rescale_factor

    # in the case total_dataset_size is None, do not do a train/test split and
    # rescale the data based on the full dataset
    if total_dataset_size is None:

        # rescale using full dataset to mean 0 and variance 1
        scaler = preprocessing.StandardScaler().fit(X_input_data)
        # equivalent to scaling gamma by 1/n_features
        X_scaled_data = scaler.transform(X_input_data)/np.sqrt(rescale)

        return (
            X_scaled_data, y_input_data,
            None, None,
            X_scaled_data, y_input_data
        )

    # stratify data such that labels appear with similar probaibility in
    # test and training
    stratify = None
    if apply_stratify:
        stratify = y_input_data

    # split data in test and train set
    X_train, X_test, y_train, y_test = train_test_split(
        X_input_data,
        y_input_data,
        train_size=(
            total_dataset_size-int(total_dataset_size*test_size_frac)
        ),
        test_size=int(total_dataset_size*test_size_frac),
        random_state=random_seed,
        stratify=stratify,
    )

    # rescale using training dataset to mean 0 and variance 1
    scaler = preprocessing.StandardScaler().fit(X_train)
    # equivalent to scaling gamma by 1/n_features
    X_scaled_train = scaler.transform(X_train)/np.sqrt(rescale)
    X_scaled_test = scaler.transform(X_test)/np.sqrt(rescale)

    # just define some variables
    X_all = np.append(X_scaled_train, X_scaled_test, axis=0)
    y_all = np.append(y_train, y_test, axis=0)

    return (
        X_scaled_train, y_train,
        X_scaled_test, y_test,
        X_all, y_all,
    )


#
# SVM fitting
# ------------
#


def fit_svm(
    train_gram,
    test_gram,
    type_dataset,
    test_size_frac,
    total_dataset_size,
    rescale_factor=1,
    n_pca_features=0,
    random_seed=1,
):
    """
    Parameters
    ----------
    train_gram : numpy.ndarray
        Precomputed training Gram matrix
    test_gram : numpy.ndarray
        Precomputed test Gram matrix
    type_dataset : int
        Dataset to use, options:
            0: breast cancer
            1: make_classification dataset
            2: circles dataset
    test_size_frac : float
        Fraction of test set in respect to total dataset
    total_dataset_size : int
        How many values to sample from dataset, 0: all
    rescale_factor : float, optional
        Additional rescale of variables, equivalent to width of Gaussian,
        large: underfitting, small: overfitting
    n_pca_features : int, optional
        If set to a number > 0, the data will be preproccesed using PCA with
        that number of principal components
    random_seed : int, optional
        Random seed for reproducibility
    """
    output = {}

    X_train, y_train, X_test, y_test, X_all, y_all = _get_data_vars(
        type_dataset,
        total_dataset_size,
        test_size_frac,
        n_pca_features,
        rescale_factor,
        random_seed,
    )

    # fit SVM
    fitted_svm = svm.SVC(kernel='precomputed', probability=True)
    fitted_svm.fit(train_gram, y_train)
    output['fitted_svm'] = fitted_svm

    y_pred_test = fitted_svm.predict(test_gram)
    y_pred_train = fitted_svm.predict(train_gram)

    # select which index are wrongly classified by predictor
    wrong_index_train = np.nonzero(np.abs(y_train-y_pred_train))[0]
    wrong_index_test = np.nonzero(np.abs(y_test-y_pred_test))[0]
    output['wrong_index_train'] = wrong_index_train
    output['wrong_index_test'] = wrong_index_test

    # get accuracy score
    output['accuracy_score'] = accuracy_score(y_test, y_pred_test)

    # ROC and AUC values
    y_pred_prob = fitted_svm.predict_proba(test_gram)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    output['roc'] = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }
    output['auc'] = auc(fpr, tpr)

    return output
