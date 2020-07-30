from __future__ import division

import json
import logging
import os
import time
from copy import deepcopy as cp

import numpy as np
from tabulate import tabulate

import additional_functions as af
import model_solvers as ms
import parallel_code as par
import prediction_solver as pred

# Load config file
root_path = os.getcwd()
config = json.loads(open(root_path + '/config').read())

logger = logging.getLogger('general_functions')
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


# MODEL SOLVER #########################################################################################################

def model_solver(model='qr', penalization='sgl', x=None, y=None, intercept=True, tol=1e-3, lambda1=None, alpha=None, index=None, tau=None, l_weights=None, gl_weights=None, parallel=True):
    """
    Receives the type of model, penalization and parameters for the model, and obtains the solutions executing the solvers from ms file
    """
    start_time = time.time()
    preprocessing_response = ms.preprocessing(model=model, penalization=penalization, x=x, y=y, intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, index=index, tau=tau, l_weights=l_weights, gl_weights=gl_weights)
    param = preprocessing_response['param']
    solution = None
    # Non penalized models
    if penalization is None and model == 'qr':
        solution = ms.qr_solver(x, y, tau=tau, intercept=intercept, tol=tol)
    elif penalization is None and model == 'lm':
        solution = ms.lm_solver(x, y, intercept=intercept, tol=tol)
    # If not parallel then call functions from ms
    if not parallel:
        if model == 'lm' and penalization == 'lasso':
            solution = ms.lm_lasso(x=x, y=y, intercept=intercept, tol=tol, param=param)
        elif model == 'lm' and penalization == 'gl':
            solution = ms.lm_gl(x=x, y=y, index=index, intercept=intercept, tol=tol, param=param)
        elif model == 'lm' and penalization == 'sgl':
            solution = ms.lm_sgl(x=x, y=y, index=index, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'lasso':
            solution = ms.qr_lasso(x=x, y=y, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'gl':
            solution = ms.qr_gl(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'sgl':
            solution = ms.qr_sgl(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'al_asgl':
            solution = ms.qr_al_asgl(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'agl_asgl':
            solution = ms.qr_agl_asgl(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'asgl':
            solution = ms.qr_asgl(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
    # If parallel then calls functions from par
    if parallel:
        if model == 'lm' and penalization == 'lasso':
            solution = par.lm_lasso_parallel(x=x, y=y, intercept=intercept, tol=tol, param=param)
        elif model == 'lm' and penalization == 'gl':
            solution = par.lm_gl_parallel(x=x, y=y, index=index, intercept=intercept, tol=tol, param=param)
        elif model == 'lm' and penalization == 'sgl':
            solution = par.lm_sgl_parallel(x=x, y=y, index=index, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'lasso':
            solution = par.qr_lasso_parallel(x=x, y=y, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'gl':
            solution = par.qr_gl_parallel(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'sgl':
            solution = par.qr_sgl_parallel(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'al_asgl':
            solution = par.qr_al_asgl_parallel(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'agl_asgl':
            solution = par.qr_agl_asgl_parallel(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
        elif model == 'qr' and penalization == 'asgl':
            solution = par.qr_asgl_parallel(x=x, y=y, index=index, tau=tau, intercept=intercept, tol=tol, param=param)
    execution_time = time.time() - start_time
    response = dict(
        parameters=preprocessing_response['parameters'],
        problem_dimension=preprocessing_response['problem_dimension'],
        solution=solution,
        execution_time=execution_time
    )
    logger.debug('Function finished without errors')
    return response


# MODEL SELECTION ######################################################################################################

def train_validate_test(x, y, index, model_param, train_pct=0.05, validate_pct=0.05, train_size=None, validate_size=None, error_type='QRE'):
    """
    Performs train/validate/test split, optimizes the parameters from the models on the validate and computes final test error
    """
    logger.debug('Starting function')
    # Split data into train/validate/test
    nrow = x.shape[0]
    data_index = np.random.choice(nrow, nrow, replace=False)
    if train_size is None:
        train_size = int(round(nrow*train_pct))
    if validate_size is None:
        validate_size = int(round(nrow*validate_pct))
    train_index = data_index[0:train_size]
    validate_index = data_index[train_size:(train_size + validate_size)]
    test_index = data_index[(train_size + validate_size):(nrow+1)]
    x_train = x[train_index, :]
    y_train = y[train_index]
    x_validate = x[validate_index, :]
    y_validate = y[validate_index]
    x_test = x[test_index, :]
    y_test = y[test_index]
    logger.debug("train_index = {}".format(train_index))
    logger.debug("validate_index = {}".format(validate_index))
    logger.debug("test_index = {}".format(test_index))
    # results is a list of length equal to the number of model penalization that we want to analyze
    results = []
    for j in range(len(model_param)):
        penalization = model_param[j]['model_solver']['penalization']
        s = penalization
        if 'asgl' in penalization:
            s += '_' + model_param[j]['weight_calculator']['weight_technique']
        logger.info('Solving model {}'.format(s))
        loop_start_time = time.time()
        # Problem dimension stores the number of different parameter values of each parameter for the models
        # vector data stores the parameters of the model in list format
        # validate results stores the best parameter values and the n-dimensional error matrix
        # test_results stores the optimal solution, the predictions and the final error
        problem_dimension = dict(n_lambda=None, n_alpha=None, n_lpw=None, n_glpw=None)
        vector_data = dict(lambda_vector=None, alpha_vector=None, l_power_weight_vector=None, gl_power_weight_vector=None)
        validate_results = dict(best_lambda=None, best_alpha=None, best_lpw=None, best_glpw=None, var_pct=None, n_trees=None, error_matrix=None)
        test_results = dict(optimal_betas=None, test_predictions=None, final_error=None)
        # Build the weight_calculator and model_solver dictionaries
        tau = None
        if model_param[j]['model_solver']['model'] == 'qr':
            tau = model_param[j]['model_solver']['tau']
        l_weights = None
        gl_weights = None
        if 'asgl' in penalization:
            # Obtain the weights in adaptive models using the weight_calculator function
            weight_calculator_dict = cp(model_param[j]['weight_calculator'])
            weight_calculator_dict.update(dict(x=x_train, y=y_train, index=index, tau=tau, penalization=penalization))
            w = af.weights_calculator(**weight_calculator_dict)
            l_weights = w['l_weights']
            gl_weights = w['gl_weights']
            vector_data.update(w['parameters'])
            problem_dimension.update(w['problem_dimension'])
            if 'pct' in weight_calculator_dict['weight_technique']:
                validate_results['var_pct'] = weight_calculator_dict['var_pct']
            if weight_calculator_dict['weight_technique'] == 'rf':
                validate_results['n_trees'] = weight_calculator_dict['n_trees']
        # Solve the model using the model_solver function
        model_solver_dict = cp(model_param[j]['model_solver'])
        model_solver_dict.update(dict(x=x_train, y=y_train, index=index, l_weights=l_weights, gl_weights=gl_weights))
        solution = model_solver(**model_solver_dict)
        solution_betas = solution["solution"]
        vector_data['lambda_vector'] = solution['parameters']['lambda_vector']
        vector_data['alpha_vector'] = solution['parameters']['alpha_vector']
        problem_dimension.update(solution['problem_dimension'])
        # Obtain the predictions and errors for the model using the validate set
        prediction = pred.predictor(intercept=model_solver_dict['intercept'], new_data=x_validate, beta_list=solution_betas)
        error_list = pred.error_calculator(true_value=y_validate, error_type=error_type, tau=tau, prediction_list=prediction)
        error_matrix = pred.error_matrix_builder(error_list=error_list, penalization=penalization, **problem_dimension)
        # Find the optima parameter values
        validate_results.update(pred.grid_optimization(error_matrix=error_matrix, vector_data=vector_data, penalization=penalization)['validate_results'])
        # Find the betas associated to the minimum error
        min_error_list = np.where(error_list == np.min(error_list))[0][0]
        optimal_betas = [solution_betas[min_error_list]]
        # Using the optimal betas obtain the final model error using the test set
        test_prediction = pred.predictor(intercept=model_solver_dict['intercept'], new_data=x_test, beta_list=optimal_betas)
        test_error = pred.error_calculator(true_value=y_test, error_type=error_type, tau=tau, prediction_list=test_prediction)[0]
        test_results['optimal_betas'] = optimal_betas
        test_results['test_predictions'] = test_prediction[0]
        test_results['final_error'] = test_error
        # Obtain l and sgl results from the first model if it is an asgl model
        if j == 0 and penalization == 'asgl':
            # Obtain model solver data
            l_model = dict(model=model_param[j]['model_solver']['model'], penalization='lasso', intercept=model_param[j]['model_solver']['intercept'], tol=model_param[j]['model_solver']['tol'], lambda1=model_param[j]['model_solver']['lambda1'], alpha=None, tau=tau, parallel=model_param[j]['model_solver']['parallel'])
            sgl_model = dict(model=model_param[j]['model_solver']['model'], penalization='sgl', intercept=model_param[j]['model_solver']['intercept'], tol=model_param[j]['model_solver']['tol'], lambda1=model_param[j]['model_solver']['lambda1'], alpha=model_param[j]['model_solver']['alpha'], tau=tau, parallel=model_param[j]['model_solver']['parallel'])
            # Obtain error matrices
            error_index = np.arange(0, len(error_list), 1)
            error_index_matrix = pred.error_matrix_builder(error_list=error_index, penalization=penalization, **problem_dimension)
            error_index_matrix_l = error_index_matrix[:, (error_index_matrix.shape[1] - 1), 0, 0]
            error_index_matrix_sgl = error_index_matrix[:, :, 0, 0]
            error_matrix_l = error_matrix[:, (error_matrix.shape[1] - 1), 0, 0]
            error_matrix_sgl = error_matrix[:, :, 0, 0]
            # Obtain optimal values
            grid_opt_l = pred.grid_optimization(error_matrix=error_matrix_l, vector_data=vector_data, penalization='lasso')
            grid_opt_sgl = pred.grid_optimization(error_matrix=error_matrix_sgl, vector_data=vector_data, penalization='sgl')
            validate_results_l = dict(best_lambda=None, best_alpha=None, best_lpw=None, best_glpw=None, var_pct=None, n_trees=None, error_matrix=None)
            validate_results_sgl = dict(best_lambda=None, best_alpha=None, best_lpw=None, best_glpw=None, var_pct=None, n_trees=None, error_matrix=None)
            validate_results_l.update(grid_opt_l['validate_results'])
            validate_results_sgl.update(grid_opt_sgl['validate_results'])
            # Obtain test results
            optimal_betas_l = [solution_betas[error_index_matrix_l[grid_opt_l['optimal_value_position'][0]]]]
            optimal_betas_sgl = [solution_betas[error_index_matrix_sgl[grid_opt_sgl['optimal_value_position'][0], grid_opt_sgl['optimal_value_position'][1]]]]
            test_prediction_l = pred.predictor(intercept=model_param[j]['model_solver']['intercept'], new_data=x_test, beta_list=optimal_betas_l)
            test_prediction_sgl = pred.predictor(intercept=model_param[j]['model_solver']['intercept'], new_data=x_test, beta_list=optimal_betas_sgl)
            test_error_l = pred.error_calculator(true_value=y_test, error_type=error_type, tau=tau, prediction_list=test_prediction_l)[0]
            test_error_sgl = pred.error_calculator(true_value=y_test, error_type=error_type, tau=tau,prediction_list=test_prediction_sgl)[0]
            test_results_l = dict(optimal_betas=optimal_betas_l, test_predictions=test_prediction_l, final_error=test_error_l)
            test_results_sgl = dict(optimal_betas=optimal_betas_sgl, test_predictions=test_prediction_sgl, final_error=test_error_sgl)
            # Append lasso results
            results.append(dict(
                model_solver=l_model,
                weight_calculator=None,
                validate_results=validate_results_l,
                test_results=test_results_l,
                set_index=dict(train_index=train_index.tolist(), validate_index=validate_index.tolist(), test_index=test_index.tolist()),
                execution_time=0))
            # Append sgl results
            results.append(dict(
                model_solver=sgl_model,
                weight_calculator=None,
                validate_results=validate_results_sgl,
                test_results=test_results_sgl,
                set_index=dict(train_index=train_index.tolist(), validate_index=validate_index.tolist(), test_index=test_index.tolist()),
                execution_time=0))
        execution_time = round(time.time() - loop_start_time,2)
        results.append(dict(
            model_solver = model_param[j]['model_solver'],
            weight_calculator = model_param[j]['weight_calculator'],
            validate_results = validate_results,
            test_results = test_results,
            set_index=dict(train_index=train_index.tolist(), validate_index=validate_index.tolist(), test_index=test_index.tolist()),
            execution_time = execution_time))
        logger.info("Model {} completed in {} seconds".format(s, execution_time))
    logger.debug('Function finished without errors')
    return results


# SIMULATOR ############################################################################################################

def automatic_simulator(data_param, model_selection_param, model_param, data_generator, n_repetitions=50, folder=''):
    """
    Receives information on the dataset to generate and the models to solve, and solves the simulations
    """
    logger.debug('Starting function')
    start_time = time.time()
    simulation_name = time.strftime('%Y-%m-%d--%H-%M-%S--', time.localtime(start_time))
    simulation_name += 'n=%d--' % data_param['n'] + 'p=%d--' % data_param['p'] + data_generator.__name__
    parameters = dict(data_param=data_param, model_selection_param=model_selection_param, data_generator=data_generator.__name__, n_repetitions=n_repetitions)
    global_results = dict()
    global_results['parameters'] = parameters
    n_models = len(model_param)
    if model_param[0]['model_solver']['penalization'] == 'asgl':
        n_models = len(model_param) + 2
    results = []
    # Generate empty lists of future outputs for each model considered in model_param
    for i in range(n_models):
        results.append(dict(
            metrics={
                'final_error': [],
                'best_lambda': [],
                'best_alpha': [],
                'best_lpw': [],
                'best_glpw': [],
                'non_zero_pred_beta': [],
                'beta_error': [],
                'correct_selection_rate': [],
                'true_positive_rate': [],
                'true_negative_rate': [],
                'f_score': []},
            summary={},
            model_solver=None,
            weight_calculator=None,
            extra_metrics={
                'index_non_zero_pred_beta': [],
                'val_non_zero_pred_beta': []
            }))
    for i in range(n_repetitions):
        # For each repetition, generate a new dataset and solve the models
        simulation_loop_timer = time.time()
        logger.info('Starting simulation {}'.format(i+1))
        # Generating synthetic data
        data = data_generator(**data_param)
        x = data['x']
        y = data['y']
        true_beta = data['true_beta']
        index = data['index']
        logger.debug('Data generated')
        # Solve the models using the train_validate_test functions
        train_validate_params = dict(x=x, y=y, index=index, model_param=model_param)
        train_validate_params.update(model_selection_param)
        tv_result = train_validate_test(**train_validate_params)
        logger.debug('Starting metrics calculation')
        # Calculate metrics for the different models
        tol = model_param[0]['model_solver']['tol']
        for j in range(n_models):
            pred_beta = tv_result[j]['test_results']['optimal_betas'][0][1:]
            index_non_zero_pred_beta = np.ndarray.astype(np.where(np.abs(pred_beta) > tol)[0], 'int')
            # noinspection PyTypeChecker
            val_non_zero_pred_beta = pred_beta[index_non_zero_pred_beta]
            bool_non_zero_pred_beta = np.abs(pred_beta) > tol
            bool_zero_pred_beta = np.abs(pred_beta) <=tol
            bool_non_zero_true_beta = np.abs(true_beta) > tol
            bool_zero_true_beta = np.abs(true_beta) <=tol
            num_non_zero_true_beta = np.sum(bool_non_zero_true_beta)
            num_zero_true_beta = np.sum(bool_zero_true_beta)
            non_zero_pos_true_beta_beta = np.where(np.abs(true_beta) > tol)[0]
            zero_pos_true_beta = np.where(np.abs(true_beta) <=tol)[0]
            true_positive = np.sum(bool_non_zero_pred_beta[non_zero_pos_true_beta_beta])
            true_negative = np.sum(bool_zero_pred_beta[zero_pos_true_beta])
            false_positive = np.sum(bool_non_zero_pred_beta[zero_pos_true_beta])
            false_negative = np.sum(bool_zero_pred_beta[non_zero_pos_true_beta_beta])
            # Check if number of non_zero or zero elts in true_beta is larger than 0 (it is used in a division)
            if num_non_zero_true_beta == 0:
                num_non_zero_true_beta = 1
            if num_zero_true_beta == 0:
                num_zero_true_beta = 1
            beta_error = np.linalg.norm(true_beta - pred_beta, 2)
            correct_selection_rate = np.sum(bool_non_zero_true_beta == bool_non_zero_pred_beta) / data_param['p']
            true_positive_rate = true_positive / num_non_zero_true_beta
            true_negative_rate = true_negative / num_zero_true_beta
            f_score = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
            results[j]['metrics']['final_error'].append(tv_result[j]['test_results']['final_error'])
            results[j]['metrics']['best_lambda'].append(tv_result[j]['validate_results']['best_lambda'])
            results[j]['metrics']['best_alpha'].append(tv_result[j]['validate_results']['best_alpha'])
            results[j]['metrics']['best_lpw'].append(tv_result[j]['validate_results']['best_lpw'])
            results[j]['metrics']['best_glpw'].append(tv_result[j]['validate_results']['best_glpw'])
            results[j]['metrics']['non_zero_pred_beta'].append(np.sum(bool_non_zero_pred_beta))
            results[j]['metrics']['beta_error'].append(beta_error)
            results[j]['metrics']['correct_selection_rate'].append(correct_selection_rate)
            results[j]['metrics']['true_positive_rate'].append(true_positive_rate)
            results[j]['metrics']['true_negative_rate'].append(true_negative_rate)
            results[j]['metrics']['f_score'].append(f_score)
            results[j]['extra_metrics']['index_non_zero_pred_beta'].append(index_non_zero_pred_beta.tolist())
            results[j]['extra_metrics']['val_non_zero_pred_beta'].append(val_non_zero_pred_beta.tolist())
            results[j]['model_solver'] = tv_result[j]['model_solver']
            results[j]['weight_calculator'] = tv_result[j]['weight_calculator']
        logger.info("Simulation {} completed in {} seconds".format(i + 1,round(time.time() - simulation_loop_timer,2)))
        # Save partial results
        global_results['results'] = results
        file_name = folder + simulation_name + '_tmp' + '.json'
        try:
            with open(file_name, 'w') as fp:
                json.dump(global_results, fp)
            logger.info('Results dump to JSON file')
        except (TypeError or IOError):
            logger.error('Error dumping results to JSON file')
    # Generate summary
    table = []
    header=None
    for j in range(n_models):
        s = results[j]['model_solver']['penalization']
        if 'asgl' in results[j]['model_solver']['penalization']:
            s += '_' + results[j]['weight_calculator']['weight_technique']
        header = ['Model']
        model_table = [s]
        for key in results[j]['metrics'].keys():
            if results[j]['metrics'][key][0] is not None and isinstance(results[j]['metrics'][key][0], (int, float)):
                # noinspection PyTypeChecker
                results[j]['summary'][key] = [round(np.mean(results[j]['metrics'][key]), 3), round(np.std(results[j]['metrics'][key]), 3)]
            else:
                results[j]['summary'][key] = [None, None]
            model_table.append(results[j]['summary'][key])
            header.append(key)
        table.append(model_table)
    logger.debug('Results summary built')
    execution_time = round(time.time() - start_time, 2)
    logger.info('All simulations completed in {} seconds'.format(execution_time))
    global_results['parameters']['execution_time'] = execution_time
    global_results['results'] = results
    file_name = folder + simulation_name + '.json'
    try:
        with open(file_name, 'w') as fp:
            json.dump(global_results, fp)
    except (TypeError or IOError):
        logger.error('Error dumping results to JSON file')
    print(tabulate(table, headers=header))
    logger.debug('Function finished without errors')
    return global_results


def automatic_analyzer(dataset, model_selection_param, model_param, n_repetitions=50, folder=''):
    """
    Receives information on the dataset to generate and the models to solve, and solves the simulations
    """
    logger.debug('Starting function')
    start_time = time.time()
    x = dataset['x']
    y = dataset['y']
    index = dataset['index']
    nrow_x = x.shape[0]
    ncol_x = x.shape[1]
    simulation_name = time.strftime('%Y-%m-%d--%H-%M-%S--', time.localtime(start_time))
    simulation_name += 'real_dataset--' + 'n=%d--' % nrow_x + 'p=%d' % ncol_x
    parameters = dict(model_selection_param=model_selection_param, n_repetitions=n_repetitions)
    global_results = dict()
    global_results['parameters'] = parameters
    n_models = len(model_param)
    if model_param[0]['model_solver']['penalization'] == 'asgl':
        n_models = len(model_param) + 2
    results = []
    # Structure of the expected results
    for i in range(n_models):
        results.append(dict(
            metrics={
                'final_error': [],
                'best_lambda': [],
                'best_alpha': [],
                'best_lpw': [],
                'best_glpw': [],
                'non_zero_pred_beta': []},
            summary={},
            model_solver=None,
            weight_calculator=None,
            extra_metrics={
                'index_non_zero_pred_beta': [],
                'val_non_zero_pred_beta': [],
                'train_index': [],
                'validate_index': [],
                'test_index': []
            }))
    for i in range(n_repetitions):
        simulation_loop_timer = time.time()
        logger.info('Starting iteration {}'.format(i))
        # Solving models using train/validate/test split
        train_validate_params = dict(x=x, y=y, index=index, model_param=model_param)
        train_validate_params.update(model_selection_param)
        tv_result = train_validate_test(**train_validate_params)
        logger.debug('Starting metrics calculation')
        # Metrics for the different models tested
        tol = model_param[0]['model_solver']['tol']
        for j in range(n_models):
            pred_beta = tv_result[j]['test_results']['optimal_betas'][0][1:]
            index_non_zero_pred_beta = np.ndarray.astype(np.where(np.abs(pred_beta) > tol)[0], 'int')
            # noinspection PyTypeChecker
            val_non_zero_pred_beta = pred_beta[index_non_zero_pred_beta]
            bool_non_zero_pred_beta = np.abs(pred_beta) > tol
            #Metrics
            results[j]['metrics']['final_error'].append(tv_result[j]['test_results']['final_error'])
            results[j]['metrics']['best_lambda'].append(tv_result[j]['validate_results']['best_lambda'])
            results[j]['metrics']['best_alpha'].append(tv_result[j]['validate_results']['best_alpha'])
            results[j]['metrics']['best_lpw'].append(tv_result[j]['validate_results']['best_lpw'])
            results[j]['metrics']['best_glpw'].append(tv_result[j]['validate_results']['best_glpw'])
            results[j]['metrics']['non_zero_pred_beta'].append(np.sum(bool_non_zero_pred_beta))
            # Extra Metrics
            results[j]['extra_metrics']['index_non_zero_pred_beta'].append(index_non_zero_pred_beta.tolist())
            results[j]['extra_metrics']['val_non_zero_pred_beta'].append(val_non_zero_pred_beta.tolist())
            results[j]['extra_metrics']['train_index'].append(tv_result[0]['set_index']['train_index'])
            results[j]['extra_metrics']['validate_index'].append(tv_result[0]['set_index']['validate_index'])
            results[j]['extra_metrics']['test_index'].append(tv_result[0]['set_index']['test_index'])
            results[j]['model_solver'] = tv_result[j]['model_solver']
            results[j]['weight_calculator'] = tv_result[j]['weight_calculator']
        logger.info("Simulation {} completed in {} seconds".format(i + 1,round(time.time() - simulation_loop_timer,2)))
        # Save partial results
        global_results['results'] = results
        file_name = folder + simulation_name + '_tmp' + '.json'
        try:
            with open(file_name, 'w') as fp:
                json.dump(global_results, fp)
            logger.info('Results dump to JSON file')
        except (TypeError or IOError):
            logger.error('Error dumping results to JSON file')
    # Generate summary
    table = []
    header=None
    for j in range(n_models):
        s = results[j]['model_solver']['penalization']
        if 'asgl' in results[j]['model_solver']['penalization']:
            # noinspection PyTypeChecker
            s += '_' + results[j]['weight_calculator']['weight_technique']
        header = ['Model']
        model_table = [s]
        for key in results[j]['metrics'].keys():
            if results[j]['metrics'][key][0] is not None and isinstance(results[j]['metrics'][key][0], (int, float)):
                # noinspection PyTypeChecker
                results[j]['summary'][key] = [round(np.mean(results[j]['metrics'][key]), 3), round(np.std(results[j]['metrics'][key]), 3)]
            else:
                results[j]['summary'][key] = [None, None]
            model_table.append(results[j]['summary'][key])
            header.append(key)
        table.append(model_table)
    logger.debug('Results summary built')
    execution_time = round(time.time() - start_time, 2)
    logger.info('All simulations completed in {} seconds'.format(execution_time))
    global_results['parameters']['execution_time'] = execution_time
    global_results['results'] = results
    file_name = folder + simulation_name + '.json'
    try:
        with open(file_name, 'w') as fp:
            json.dump(global_results, fp)
    except (TypeError or IOError):
        logger.error('Error dumping results to JSON file')
    print(tabulate(table, headers=header))
    logger.debug('Function finished without errors')
    return global_results


########################################################################################################################
