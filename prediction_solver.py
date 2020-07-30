from __future__ import division

import json
import logging
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

# Load config file
root_path = os.getcwd()
config = json.loads(open(root_path + '/config').read())

logger = logging.getLogger('prediction_solver')
logging_level = logging.getLevelName(config["logger"]["loggingLevel"])
stream_handler_logging_level = logging.getLevelName(config["logger"]["stream_handler_logging_level"])
file_handler_logging_level = logging.getLevelName(config["logger"]["file_handler_logging_level"])
logger.setLevel(logging_level)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

if config["logger"]['logger_output'] == 'stream':
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_handler_logging_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
elif config["logger"]['logger_output'] == 'file':
    file_handler = logging.FileHandler(config["logger"]["file_name"])
    file_handler.setLevel(file_handler_logging_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# PREDICTION FUNCTIONS #################################################################################################

def predictor(intercept, new_data, beta_list):
    """
    Receives a list of beta solutions for a regression model and a matrix of regressors and computes the predicted value
    """
    # Check that the new data is an array and that the beta_list is a list
    if not isinstance(new_data, np.ndarray) or \
            not isinstance(beta_list, list):
        logger.error('incorrect input parameter')
        raise ValueError('incorrect input parameters')
    # If we have intercept in the model, we add a column of ones to x so solution dimension and x dimension matches
    if intercept:
        new_data = np.c_[np.ones(new_data.shape[0]), new_data]
    new_data_n_betas = new_data.shape[1]
    n_betas = len(beta_list[0])
    if n_betas != new_data_n_betas:
        logger.error('Model dimension and new data dimension does not match')
        raise ValueError('Model dimension and new data dimension does not match')
    # We store the predictions in a list
    prediction_list = []
    for elt in beta_list:
        prediction_list.append(np.dot(new_data, elt))
    logger.debug('Function finished without errors')
    return prediction_list


def error_calculator(true_value, error_type="MSE", tau=None, prediction_list=None):
    """
    Computes the error between the predicted value and the true value of the response variable
    """
    # Check that the true_value (response) is an array, prediction_list is a list and error_type is a string
    valid_error_types = ["MSE", "MAE", "MDAE", "QRE"]
    if not isinstance(true_value, np.ndarray) or \
            not isinstance(prediction_list, list) or\
            not isinstance(error_type, str):
        logger.error('incorrect input parameters')
        raise ValueError('incorrect input parameters')
    # Check that the error_type is a valid error type considered
    if error_type not in valid_error_types:
        raise ValueError('invalid error type')
    n_true_value = true_value.shape[0]
    n_predictions = len(prediction_list[0])
    if n_true_value != n_predictions:
        logger.error('Dimension of test data does not match dimension of prediction')
        raise ValueError('Dimension of test data does not match dimension of prediction')
    # For each prediction, we store the error associated to that prediction in a list
    error_list = []
    if error_type == "MSE":
        for elt in prediction_list:
            error_list.append(mean_squared_error(true_value, elt))
    elif error_type == "MAE":
        for elt in prediction_list:
            error_list.append(mean_absolute_error(true_value, elt))
    elif error_type == "MDAE":
        for elt in prediction_list:
            error_list.append(median_absolute_error(true_value, elt))
    elif error_type == "QRE":
        for elt in prediction_list:
            error_list.append((1.0/len(true_value))*np.sum(0.5*np.abs(true_value-elt)+(tau-0.5)*(true_value-elt)))
    else:
        raise ValueError('Unable to calculate error')
    logger.debug('Function finished without errors')
    return error_list


def error_matrix_builder(error_list, penalization, n_lambda=None, n_alpha=None, n_lpw=None, n_glpw=None):
    """
    Receives a list of error values and the dimension of the parameters from the models,  and based on the way the
    parameters are ordered into tuples in the preprocessing function, we create a n-dimensional error matrix
    """
    if penalization == 'lasso' or penalization == 'gl':
        error_matrix = np.asarray(error_list).reshape(n_lambda)
    elif penalization == 'sgl':
        error_matrix = np.asarray(error_list).reshape((n_lambda, n_alpha))
    elif penalization == 'al_asgl':
        error_matrix = np.asarray(error_list).reshape((n_lambda, n_alpha, n_lpw))
    elif penalization == 'agl_asgl':
        error_matrix = np.asarray(error_list).reshape((n_lambda, n_alpha, n_glpw))
    elif penalization == 'asgl':
        error_matrix = np.asarray(error_list).reshape((n_lambda, n_alpha, n_lpw, n_glpw))
    else:
        error_matrix = None
        logger.error('Error computing constructing the error matrix')
    logger.debug('Function finished without errors')
    return error_matrix


# MODEL SELECTION ######################################################################################################

def grid_optimization(error_matrix, vector_data, penalization):
    """
    Receives a error_matrix and indicates the position of the matrix in which the global minimum is achieved
    and the optimal values in which for which that minimum is achieved
    """
    best_performance = np.where(error_matrix == np.min(error_matrix))
    optimal_values = []
    for elt in best_performance:
        optimal_values.append(elt[0])
    response = dict(optimal_value_position=optimal_values,
                    validate_results=dict(best_lambda=None, best_alpha=None, best_lpw=None, best_glpw=None, error_matrix=error_matrix))
    response['validate_results']['best_lambda'] = vector_data['lambda_vector'][optimal_values[0]]
    if penalization == 'sgl' or "asgl" in penalization:
        response['validate_results']['best_alpha'] = vector_data['alpha_vector'][optimal_values[1]]
    if penalization == 'al_asgl':
        response['validate_results']['best_lpw'] = vector_data['l_power_weight_vector'][optimal_values[2]]
    if penalization == 'agl_asgl':
        response['validate_results']['best_glpw'] = vector_data['gl_power_weight_vector'][optimal_values[2]]
    if penalization == 'asgl':
        response['validate_results']['best_lpw'] = vector_data['l_power_weight_vector'][optimal_values[2]]
        response['validate_results']['best_glpw'] = vector_data['gl_power_weight_vector'][optimal_values[3]]
    logger.debug('Function finished without errors')
    return response


########################################################################################################################
