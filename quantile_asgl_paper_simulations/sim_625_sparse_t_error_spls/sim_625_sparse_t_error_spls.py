from __future__ import division

import json
import logging
import os
import time
from copy import deepcopy as cp

import numpy as np

import general_functions as gf
import numeric_data_generator as ngen
import prediction_solver as pred
from tabulate import tabulate

"""
n=5200--p=625--sparse groups--dg_t--spls

El Sparse PLS no esta implementado en python por lo que tendremos que usar la implementacion de R, y resolver los
modelos en python
"""

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

# INPUT DE PARAMETROS ##################################################################################################

# Model solver
model = 'qr'
penalization = 'asgl'
intercept = True
tol = 1e-4
parallel = True
lambda1 = (10.0**np.arange(-3, 1.01, 0.2)).tolist()
alpha = (np.r_[np.arange(0.0, 0.3, 0.03), np.arange(0.3, 0.7, 0.1), np.arange(0.7, 0.99, 0.03), np.array([1])]).tolist()
tau = 0.5

# Weight calculator
l_power_weight = np.arange(0, 1.41, 0.2).tolist()
gl_power_weight = np.arange(0, 1.41, 0.2).tolist()
var_pct = 0.8
weight_tol = 1e-4

# Model_selection_param
train_size = 100
validate_size = 100
error_type = 'QRE'

# Automatic simulator
n_repetitions = 50
folder = ''

model_selection_param = dict(train_size=train_size, validate_size=validate_size, error_type=error_type)
asgl_wc_spls = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, weight_technique='spls', spls_param=dict(eta=0.5, k=30))
asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)
model_param = [dict(model_solver=asgl_model, weight_calculator=asgl_wc_spls)]

########################################################################################################################
"""
 FASE 1: CONSTRUIR LOS DATAFRAMES Y ALMACENARLOS
"""

save_data_route = 'analysis/quantile_asgl_paper_simulations/sim_625_sparse_t_error_spls/sim_625_sparse_t_error_spls_data/'
data_param = dict(n=5200, num_groups=25, group_size=25, p=625, ro=0.5, num_non_zero_coef_per_group=8, num_non_zero_groups=7)
data_generator = ngen.dg_t

for i in range(n_repetitions):
    data = data_generator(**data_param)
    x = data['x']
    y = data['y']
    index = data['index']
    true_beta = data['true_beta']
    np.savetxt(save_data_route + str(i) + "_x.txt", x, delimiter=",")
    np.savetxt(save_data_route + str(i) + "_y.txt", y, delimiter=",")
    np.savetxt(save_data_route + str(i) + "_index.txt", index, delimiter=",")
    np.savetxt(save_data_route + str(i) + "_true_beta.txt", true_beta, delimiter=",")

    # STORE TRAIN VALIDATE TEST INDEX
    nrow = x.shape[0]
    data_index = np.random.choice(nrow, nrow, replace=False)
    train_index = data_index[0:train_size]
    validate_index = data_index[train_size:(train_size + validate_size)]
    test_index = data_index[(train_size + validate_size):(nrow+1)]
    np.savetxt(save_data_route + str(i) + "_train_index.txt", train_index, delimiter=",")
    np.savetxt(save_data_route + str(i) + "_validate_index.txt", validate_index, delimiter=",")
    np.savetxt(save_data_route + str(i) + "_test_index.txt", test_index, delimiter=",")

########################################################################################################################
"""
FASE 2: RESOLVER LOS MODELOS
"""

# CARGAR LOS DATOS #####################################################################################################

save_data_route = 'analysis/quantile_asgl_paper_simulations/sim_625_sparse_t_error_spls/sim_625_sparse_t_error_spls_data/'

# RESOLUCION DEL MODELO
logger.debug('Starting function')
start_time = time.time()
simulation_name = time.strftime('%Y-%m-%d--%H-%M-%S--', time.localtime(start_time))
simulation_name += 'n=%d--' % data_param['n'] + 'p=%d--' % data_param['p'] + data_generator.__name__
parameters = dict(data_param=data_param, model_selection_param=model_selection_param,
                  data_generator=data_generator.__name__, n_repetitions=n_repetitions)
global_results = dict()
global_results['parameters'] = parameters
n_models = len(model_param)
if model_param[0]['model_solver']['penalization'] == 'asgl':
    n_models = len(model_param) + 2
results_automatic_simulator = []
# Generate empty lists of future outputs for each model considered in model_param
for i in range(n_models):
    results_automatic_simulator.append(dict(
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
    x = np.loadtxt(save_data_route + str(i) + '_x.txt', delimiter=',')
    y = np.loadtxt(save_data_route + str(i) + '_y.txt', delimiter=',')
    index = np.loadtxt(save_data_route + str(i) + '_index.txt', delimiter=',')
    true_beta = np.loadtxt(save_data_route + str(i) + '_true_beta.txt', delimiter=',')
    train_index = np.loadtxt(save_data_route + str(i) + '_train_index.txt', delimiter=',')
    validate_index = np.loadtxt(save_data_route + str(i) + '_validate_index.txt', delimiter=',')
    test_index = np.loadtxt(save_data_route + str(i) + '_test_index.txt', delimiter=',')
    tmp_weight = np.loadtxt(save_data_route + str(i) + '_tmp_weight.txt', delimiter=',')

    index = index.astype(int)
    true_beta = true_beta.astype(int)
    train_index = train_index.astype(int)
    validate_index = validate_index.astype(int)
    test_index = test_index.astype(int)

    x_train = x[train_index, :]
    y_train = y[train_index]
    x_validate = x[validate_index, :]
    y_validate = y[validate_index]
    x_test = x[test_index, :]
    y_test = y[test_index]
    # For each repetition, generate a new dataset and solve the models
    simulation_loop_timer = time.time()
    logger.info('Starting simulation {}'.format(i + 1))
    # FRAGMENTO DE TRAIN VALIDATE TEST
    results_tv = []
    for j in range(len(model_param)):
        penalization = model_param[j]['model_solver']['penalization']
        s = penalization
        s += '_' + 'spls'
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
        # IMPORTAR LOS PESOS DESDE ARCHIVO EXTERNO
        tmp_weight[tmp_weight < weight_tol] = weight_tol
        l_weights = []
        n_lpw = len(l_power_weight)
        for lpw in l_power_weight:
            l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        unique_index = np.unique(index)
        gl_weights = []
        n_glpw = len(gl_power_weight)
        for glpw in gl_power_weight:
            tmp_list = []
            for i in range(len(unique_index)):
                tmp_list.append(1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
            gl_weights.append(tmp_list)
        w = dict(
            problem_dimension=dict(n_lpw=n_lpw, n_glpw=n_glpw),
            parameters=dict(l_power_weight_vector=l_power_weight, gl_power_weight_vector=gl_power_weight, n_trees=None, var_pct=0),
            l_weights=l_weights, gl_weights=gl_weights)
        # FIN DE: IMPORTAR LOS PESOS DESDE ARCHIVO EXTERNO
        l_weights = w['l_weights']
        gl_weights = w['gl_weights']
        vector_data.update(w['parameters'])
        problem_dimension.update(w['problem_dimension'])
        validate_results['var_pct'] = 0
        # Solve the model using the model_solver function
        model_solver_dict = cp(model_param[j]['model_solver'])
        model_solver_dict.update(dict(x=x_train, y=y_train, index=index, l_weights=l_weights, gl_weights=gl_weights))
        solution = gf.model_solver(**model_solver_dict)
        solution_betas = solution["solution"]
        vector_data['lambda_vector'] = solution['parameters']['lambda_vector']
        vector_data['alpha_vector'] = solution['parameters']['alpha_vector']
        problem_dimension.update(solution['problem_dimension'])
        # Obtain the predictions and errors for the model using the validate set
        prediction = pred.predictor(intercept=model_solver_dict['intercept'], new_data=x_validate, beta_list=solution_betas)
        error_list = pred.error_calculator(true_value=y_validate, error_type=error_type, tau=tau,
                                           prediction_list=prediction)
        error_matrix = pred.error_matrix_builder(error_list=error_list, penalization=penalization, **problem_dimension)
        # Find the optima parameter values
        validate_results.update(
            pred.grid_optimization(error_matrix=error_matrix, vector_data=vector_data, penalization=penalization)[
                'validate_results'])
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
            l_model = dict(model=model_param[j]['model_solver']['model'], penalization='lasso',
                           intercept=model_param[j]['model_solver']['intercept'], tol=model_param[j]['model_solver']['tol'],
                           lambda1=model_param[j]['model_solver']['lambda1'], alpha=None, tau=tau,
                           parallel=model_param[j]['model_solver']['parallel'])
            sgl_model = dict(model=model_param[j]['model_solver']['model'], penalization='sgl',
                             intercept=model_param[j]['model_solver']['intercept'],
                             tol=model_param[j]['model_solver']['tol'], lambda1=model_param[j]['model_solver']['lambda1'],
                             alpha=model_param[j]['model_solver']['alpha'], tau=tau,
                             parallel=model_param[j]['model_solver']['parallel'])
            # Obtain error matrices
            error_index = np.arange(0, len(error_list), 1)
            error_index_matrix = pred.error_matrix_builder(error_list=error_index, penalization=penalization,
                                                           **problem_dimension)
            error_index_matrix_l = error_index_matrix[:, (error_index_matrix.shape[1] - 1), 0, 0]
            error_index_matrix_sgl = error_index_matrix[:, :, 0, 0]
            error_matrix_l = error_matrix[:, (error_matrix.shape[1] - 1), 0, 0]
            error_matrix_sgl = error_matrix[:, :, 0, 0]
            # Obtain optimal values
            grid_opt_l = pred.grid_optimization(error_matrix=error_matrix_l, vector_data=vector_data, penalization='lasso')
            grid_opt_sgl = pred.grid_optimization(error_matrix=error_matrix_sgl, vector_data=vector_data,
                                                  penalization='sgl')
            validate_results_l = dict(best_lambda=None, best_alpha=None, best_lpw=None, best_glpw=None, var_pct=None,
                                      n_trees=None, error_matrix=None)
            validate_results_sgl = dict(best_lambda=None, best_alpha=None, best_lpw=None, best_glpw=None, var_pct=None,
                                        n_trees=None, error_matrix=None)
            validate_results_l.update(grid_opt_l['validate_results'])
            validate_results_sgl.update(grid_opt_sgl['validate_results'])
            # Obtain test results
            optimal_betas_l = [solution_betas[error_index_matrix_l[grid_opt_l['optimal_value_position'][0]]]]
            optimal_betas_sgl = [solution_betas[error_index_matrix_sgl[
                grid_opt_sgl['optimal_value_position'][0], grid_opt_sgl['optimal_value_position'][1]]]]
            test_prediction_l = pred.predictor(intercept=model_param[j]['model_solver']['intercept'], new_data=x_test,
                                               beta_list=optimal_betas_l)
            test_prediction_sgl = pred.predictor(intercept=model_param[j]['model_solver']['intercept'], new_data=x_test,
                                                 beta_list=optimal_betas_sgl)
            test_error_l = pred.error_calculator(true_value=y_test, error_type=error_type, tau=tau, prediction_list=test_prediction_l)[0]
            test_error_sgl = pred.error_calculator(true_value=y_test, error_type=error_type, tau=tau, prediction_list=test_prediction_sgl)[0]
            test_results_l = dict(optimal_betas=optimal_betas_l, test_predictions=test_prediction_l,
                                  final_error=test_error_l)
            test_results_sgl = dict(optimal_betas=optimal_betas_sgl, test_predictions=test_prediction_sgl,
                                    final_error=test_error_sgl)
            # Append lasso results
            results_tv.append(dict(
                model_solver=l_model,
                weight_calculator=None,
                validate_results=validate_results_l,
                test_results=test_results_l,
                set_index=dict(train_index=train_index.tolist(), validate_index=validate_index.tolist(),
                               test_index=test_index.tolist()),
                execution_time=0))
            # Append sgl results
            results_tv.append(dict(
                model_solver=sgl_model,
                weight_calculator=None,
                validate_results=validate_results_sgl,
                test_results=test_results_sgl,
                set_index=dict(train_index=train_index.tolist(), validate_index=validate_index.tolist(),
                               test_index=test_index.tolist()),
                execution_time=0))
        execution_time = round(time.time() - loop_start_time, 2)
        results_tv.append(dict(
            model_solver=model_param[j]['model_solver'],
            weight_calculator=model_param[j]['weight_calculator'],
            validate_results=validate_results,
            test_results=test_results,
            set_index=dict(train_index=train_index.tolist(), validate_index=validate_index.tolist(),
                           test_index=test_index.tolist()),
            execution_time=execution_time))
        logger.info("Model {} completed in {} seconds".format(s, execution_time))
    logger.debug('Function finished without errors')
    # FIN DE FRAGMENTO DE TRAIN VALIDATE TEST
    tv_result = cp(results_tv)
    # Calculate metrics for the different models
    tol = model_param[0]['model_solver']['tol']
    for j in range(n_models):
        pred_beta = tv_result[j]['test_results']['optimal_betas'][0][1:]
        index_non_zero_pred_beta = np.ndarray.astype(np.where(np.abs(pred_beta) > tol)[0], 'int')
        # noinspection PyTypeChecker
        val_non_zero_pred_beta = pred_beta[index_non_zero_pred_beta]
        bool_non_zero_pred_beta = np.abs(pred_beta) > tol
        bool_zero_pred_beta = np.abs(pred_beta) <= tol
        bool_non_zero_true_beta = np.abs(true_beta) > tol
        bool_zero_true_beta = np.abs(true_beta) <= tol
        num_non_zero_true_beta = np.sum(bool_non_zero_true_beta)
        num_zero_true_beta = np.sum(bool_zero_true_beta)
        non_zero_pos_true_beta_beta = np.where(np.abs(true_beta) > tol)[0]
        zero_pos_true_beta = np.where(np.abs(true_beta) <= tol)[0]
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
        results_automatic_simulator[j]['metrics']['final_error'].append(tv_result[j]['test_results']['final_error'])
        results_automatic_simulator[j]['metrics']['best_lambda'].append(tv_result[j]['validate_results']['best_lambda'])
        results_automatic_simulator[j]['metrics']['best_alpha'].append(tv_result[j]['validate_results']['best_alpha'])
        results_automatic_simulator[j]['metrics']['best_lpw'].append(tv_result[j]['validate_results']['best_lpw'])
        results_automatic_simulator[j]['metrics']['best_glpw'].append(tv_result[j]['validate_results']['best_glpw'])
        results_automatic_simulator[j]['metrics']['non_zero_pred_beta'].append(np.sum(bool_non_zero_pred_beta))
        results_automatic_simulator[j]['metrics']['beta_error'].append(beta_error)
        results_automatic_simulator[j]['metrics']['correct_selection_rate'].append(correct_selection_rate)
        results_automatic_simulator[j]['metrics']['true_positive_rate'].append(true_positive_rate)
        results_automatic_simulator[j]['metrics']['true_negative_rate'].append(true_negative_rate)
        results_automatic_simulator[j]['metrics']['f_score'].append(f_score)
        results_automatic_simulator[j]['extra_metrics']['index_non_zero_pred_beta'].append(index_non_zero_pred_beta.tolist())
        results_automatic_simulator[j]['extra_metrics']['val_non_zero_pred_beta'].append(val_non_zero_pred_beta.tolist())
        results_automatic_simulator[j]['model_solver'] = tv_result[j]['model_solver']
        results_automatic_simulator[j]['weight_calculator'] = tv_result[j]['weight_calculator']
    logger.info("Simulation {} completed in {} seconds".format(i + 1, round(time.time() - simulation_loop_timer, 2)))
    # Save partial results
    global_results['results'] = results_automatic_simulator
    file_name = folder + simulation_name + '_tmp' + '.json'
    try:
        with open(file_name, 'w') as fp:
            json.dump(global_results, fp)
        logger.info('Results dump to JSON file')
    except (TypeError or IOError):
        logger.error('Error dumping results to JSON file')
    # Generate summary
table = []
header = None
for j in range(n_models):
    s = results_automatic_simulator[j]['model_solver']['penalization']
    if 'asgl' in results_automatic_simulator[j]['model_solver']['penalization']:
        s += '_' + results_automatic_simulator[j]['weight_calculator']['weight_technique']
    header = ['Model']
    model_table = [s]
    for key in results_automatic_simulator[j]['metrics'].keys():
        if results_automatic_simulator[j]['metrics'][key][0] is not None and isinstance(results_automatic_simulator[j]['metrics'][key][0], (int, float)):
            # noinspection PyTypeChecker
            results_automatic_simulator[j]['summary'][key] = [round(np.mean(results_automatic_simulator[j]['metrics'][key]), 3),
                                          round(np.std(results_automatic_simulator[j]['metrics'][key]), 3)]
        else:
            results_automatic_simulator[j]['summary'][key] = [None, None]
        model_table.append(results_automatic_simulator[j]['summary'][key])
        header.append(key)
    table.append(model_table)
logger.debug('Results summary built')
execution_time = round(time.time() - start_time, 2)
logger.info('All simulations completed in {} seconds'.format(execution_time))
global_results['parameters']['execution_time'] = execution_time
global_results['results'] = results_automatic_simulator
file_name = folder + simulation_name + '.json'
try:
    with open(file_name, 'w') as fp:
        json.dump(global_results, fp)
except (TypeError or IOError):
    logger.error('Error dumping results to JSON file')
print(tabulate(table, headers=header))
logger.debug('Function finished without errors')