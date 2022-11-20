import sys, os
import numpy as np
from joblib import load

from mlops.utils import get_all_h_param_comb, tune_and_save, train_dev_test_split, preprocess_digits
from sklearn import svm, metrics, datasets

# test case to check if all the combinations of the hyper parameters are indeed getting created
def test_get_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)

def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)
    return h_param_comb

def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train

def test_tune_and_save():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)

    assert actual_model_path == model_path
    assert os.path.exists(actual_model_path)
    assert type(load(actual_model_path)) == type(clf)


def test_not_biased():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert len(set(predicted))!=1


def test_predicts_all():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert set(predicted) == set(y_test)

# test to check if the randomized state are same
def test_check_random_state_equal():
    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    x_t=[]
    random_state = 0
    for i in range(2):
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
            data, label, train_frac, dev_frac, random_state)
        x_t.append(y_train)
    assert len(x_t[0]) == len(x_t[1])

# test to check if the randomized state are same
def test_check_random_state_not_equal():
    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    x_t=[]
    y_t=[]
    random_state = 10
    for i in range(2):
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
            data, label, train_frac, dev_frac, random_state)
        x_t.append(x_train)
        y_t.append(y_train)
    assert len(x_t[0]) == len(x_t[1])
    assert len(y_t[0]) == len(y_t[1])