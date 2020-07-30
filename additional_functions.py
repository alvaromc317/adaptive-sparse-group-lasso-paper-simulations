from __future__ import division

import json
import logging
import os
from multiprocessing import cpu_count

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate
from sklearn.decomposition import SparsePCA
import model_solvers as ms

# Load config file
root_path = os.getcwd()
config = json.loads(open(root_path + '/config').read())

logger = logging.getLogger('additional_functions')
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


# WEIGHT CALCULATION ###################################################################################################

def weights_calculator(x, y, penalization=None, index=None, tau=None, weight_technique=None, weight_tol=1e-4, l_power_weight=None, gl_power_weight=None, var_pct=None, n_trees=None, spca_param=None):
    """
    Calculates lists of weights for the adaptive models
    """
    # If l_power_weight is a number, transform it into a list of one element
    # Check that var_pct and n_trees are numbers
    if l_power_weight is not None and isinstance(l_power_weight, (np.float, np.int)):
        l_power_weight = [l_power_weight]
    if gl_power_weight is not None and isinstance(gl_power_weight, (np.float, np.int)):
        gl_power_weight = [gl_power_weight]
    if var_pct is not None and isinstance(var_pct, (list, np.ndarray)):
        raise ValueError('Invalid var_pct value')
    if n_trees is not None and isinstance(n_trees, (list, np.ndarray)):
        raise ValueError('Invalid n_trees value')
    if 'al' in penalization:
        gl_power_weight = None
    elif 'agl' in penalization:
        l_power_weight = None
    # Stores empty variables
    n_lpw = None
    n_glpw = None
    l_weights = None
    gl_weights = None
    # Obtain the first principal component
    if weight_technique == 'pca_1':
        pca = PCA(n_components=1)
        pca.fit(x)
        tmp_weight = np.abs(pca.components_).flatten()
        # If the value of the component is close to zero, we set it as 1e-4 so that we can calculate 1/this value
        tmp_weight[tmp_weight < weight_tol] = weight_tol
        # We obtain the adaptive lasso weights as 1/tmp_weight
        if penalization == 'al_asgl' or penalization == 'asgl':
            l_weights = []
            n_lpw = len(l_power_weight)
            for lpw in l_power_weight:
                l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        if penalization == 'agl_asgl' or penalization == 'asgl':
            unique_index = np.unique(index)
            gl_weights = []
            n_glpw = len(gl_power_weight)
            for glpw in gl_power_weight:
                tmp_list = []
                for i in range(len(unique_index)):
                    tmp_list.append(1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
                gl_weights.append(tmp_list)
    # Use enough PCs to recover a specified % of variability
    elif weight_technique == 'pca_d':
        # If var_pct is equal to one, the algorithm selects just 1 component, not 100% of the variability.
        if var_pct == 1:
            var_pct2 = np.min((x.shape[0], x.shape[1]))
        else:
            var_pct2 = var_pct
        pca = PCA(n_components=var_pct2)
        # t is the matrix of "scores" (the projection of x into the PC subspace)
        # p is the matrix of "loadings" (the PCs, the eigen vectors)
        t = pca.fit_transform(x)
        p = pca.components_.T
        # Solve an unpenalized qr model using the obtained PCs
        beta_qr = ms.qr_solver(x=t, y=y, tau=tau, intercept=False)[0]
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p, beta_qr.T).flatten())
        # If the value of the component is close to zero, we set it as 1e-4 so that we can calculate 1/this value
        tmp_weight[tmp_weight < weight_tol] = weight_tol
        # We obtain the adaptive lasso weights as 1/tmp_weight
        if penalization == 'al_asgl' or penalization == 'asgl':
            l_weights = []
            n_lpw = len(l_power_weight)
            for lpw in l_power_weight:
                l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        if penalization == 'agl_asgl' or penalization == 'asgl':
            unique_index = np.unique(index)
            gl_weights = []
            n_glpw = len(gl_power_weight)
            for glpw in gl_power_weight:
                tmp_list = []
                for i in range(len(unique_index)):
                    tmp_list.append(1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
                gl_weights.append(tmp_list)
    # Obtain the first PLS component
    elif weight_technique == 'pls_1':
        # x_loadings_ is the pls equivalent to the PCs
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(x, y)
        tmp_weight = np.abs(pls.x_rotations_).flatten()
        # If the value of the component is close to zero, we set it as 1e-4 so that we can calculate 1/this value
        tmp_weight[tmp_weight < weight_tol] = weight_tol
        # We obtain the adaptive lasso weights as 1/tmp_weight
        if penalization == 'al_asgl' or penalization == 'asgl':
            l_weights = []
            n_lpw = len(l_power_weight)
            for lpw in l_power_weight:
                l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        if penalization == 'agl_asgl' or penalization == 'asgl':
            unique_index = np.unique(index)
            gl_weights = []
            n_glpw = len(gl_power_weight)
            for glpw in gl_power_weight:
                tmp_list = []
                for i in range(len(unique_index)):
                    tmp_list.append(1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
                gl_weights.append(tmp_list)
    # Use PLS to recover a specified % of variability. Take into account that PLS do not recover the total variability
    # of x so it is possible not to reach the specified % of variability
    elif weight_technique == 'pls_d':
        total_variance_in_x = np.sum(np.var(x, axis=0))
        pls = PLSRegression(n_components=np.min((x.shape[0], x.shape[1])), scale=False)
        pls.fit(x, y)
        variance_in_pls = np.var(pls.x_scores_, axis=0)
        fractions_of_explained_variance = np.cumsum(variance_in_pls / total_variance_in_x)
        possible_var_explained = np.min((var_pct, np.max(fractions_of_explained_variance)))
        var_pct = np.min((var_pct, np.max(possible_var_explained)))
        n_comp = np.argmax(fractions_of_explained_variance >= possible_var_explained) + 1
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(x, y)
        tmp_weight = np.abs(np.asarray(pls.coef_).flatten())
        # If the value of the component is close to zero, we set it as 1e-4 so that we can calculate 1/this value
        tmp_weight[tmp_weight < weight_tol] = weight_tol
        # We obtain the adaptive lasso weights as 1/tmp_weight
        if penalization == 'al_asgl' or penalization == 'asgl':
            l_weights = []
            n_lpw = len(l_power_weight)
            for lpw in l_power_weight:
                l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        if penalization == 'agl_asgl' or penalization == 'asgl':
            unique_index = np.unique(index)
            gl_weights = []
            n_glpw = len(gl_power_weight)
            for glpw in gl_power_weight:
                tmp_list = []
                for i in range(len(unique_index)):
                    tmp_list.append(1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
                gl_weights.append(tmp_list)
    # Use random forest to obtain probabilities of appearance of each variable
    elif weight_technique == 'rf':
        # n_jobs define the number of tasks to solve in parallel
        n_jobs = np.min([config['max_cores'], cpu_count()])
        rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=n_jobs)
        rf.fit(x, y)
        tmp_weight = rf.feature_importances_
        # If the value of the component is close to zero, we set it as 1e-4 so that we can calculate 1/this value
        tmp_weight[np.abs(tmp_weight) < weight_tol] = weight_tol
        # We obtain the adaptive lasso weights as 1/tmp_weight
        if penalization == 'al_asgl' or penalization == 'asgl':
            l_weights = []
            n_lpw = len(l_power_weight)
            for lpw in l_power_weight:
                l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        if penalization == 'agl_asgl' or penalization == 'asgl':
            unique_index = np.unique(index)
            gl_weights = []
            n_glpw = len(gl_power_weight)
            for glpw in gl_power_weight:
                tmp_list = []
                for i in range(len(unique_index)):
                    tmp_list.append(
                        1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
                gl_weights.append(tmp_list)
    # Use a non-penalized model (only for n>p cases)
    elif weight_technique == 'unpenalized_qr':
        # Solve an unpenalized qr model
        tmp_weight = ms.qr_solver(x=x, y=y, tau=tau, intercept=False)[0]
        # If the value of the component is close to zero, we set it as 1e-4 so that we can calculate 1/this value
        tmp_weight[tmp_weight < weight_tol] = weight_tol
        # We obtain the adaptive lasso weights as 1/tmp_weight
        if penalization == 'al_asgl' or penalization == 'asgl':
            l_weights = []
            n_lpw = len(l_power_weight)
            for lpw in l_power_weight:
                l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        if penalization == 'agl_asgl' or penalization == 'asgl':
            unique_index = np.unique(index)
            gl_weights = []
            n_glpw = len(gl_power_weight)
            for glpw in gl_power_weight:
                tmp_list = []
                for i in range(len(unique_index)):
                    tmp_list.append(1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
                gl_weights.append(tmp_list)
    elif weight_technique == 'spca_d':
        x_center = x - x.mean(axis=0)
        alpha_spca = spca_param['alpha']
        ridge_alpha = spca_param['ridge_alpha']
        total_variance_in_x = np.sum(np.var(x, axis=0))
        spca = SparsePCA(n_components=np.min((x.shape[0], x.shape[1])), alpha=alpha_spca, ridge_alpha=ridge_alpha, normalize_components=True)
        # Obtain explained variance using spca
        t = spca.fit_transform(x_center)
        p = spca.components_.T
        t_spca_qr = np.linalg.qr(t)
        # QR decomposition of modified PCs
        r_spca = t_spca_qr[1]
        t_spca_variance = np.diag(r_spca) ** 2 / x.shape[0]
        # compute variance_ratio
        fractions_of_explained_variance = np.cumsum(t_spca_variance / total_variance_in_x)
        possible_var_explained = np.min((var_pct, np.max(fractions_of_explained_variance)))
        var_pct = np.min((var_pct, np.max(possible_var_explained)))
        n_comp = np.argmax(fractions_of_explained_variance >= possible_var_explained) + 1
        beta_qr = ms.qr_solver(x=t[:,0:n_comp], y=y, tau=tau, intercept=False)[0]
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p[:,0:n_comp], beta_qr.T).flatten())
        # If the value of the component is close to zero, we set it as 1e-4 so that we can calculate 1/this value
        tmp_weight[tmp_weight < weight_tol] = weight_tol
        # We obtain the adaptive lasso weights as 1/tmp_weight
        if penalization == 'al_asgl' or penalization == 'asgl':
            l_weights = []
            n_lpw = len(l_power_weight)
            for lpw in l_power_weight:
                l_weights.append((1.0 / tmp_weight) ** lpw)
        # We obtain the gl weights as 1/l2 norm of the weights associated to that group
        if penalization == 'agl_asgl' or penalization == 'asgl':
            unique_index = np.unique(index)
            gl_weights = []
            n_glpw = len(gl_power_weight)
            for glpw in gl_power_weight:
                tmp_list = []
                for i in range(len(unique_index)):
                    tmp_list.append(1.0 / np.linalg.norm(tmp_weight[np.where(index == unique_index[i])[0]], 2) ** glpw)
                gl_weights.append(tmp_list)
    response = dict(
        problem_dimension=dict(n_lpw=n_lpw, n_glpw=n_glpw),
        parameters=dict(l_power_weight_vector=l_power_weight, gl_power_weight_vector=gl_power_weight, n_trees=n_trees, var_pct=var_pct),
        l_weights=l_weights, gl_weights=gl_weights)
    logger.debug('Function finished without errors')
    return response


# INDEX CALCULATOR #####################################################################################################

def pca_group_creator(x):
    """
    Create groups for the columns of x based on the PC in which each variable maximizes its weight
    """
    # Obtain as much PCs as possible
    p = x.shape[1]
    pca = PCA()
    pca.fit(x)
    prin_comp = pca.components_.T
    group_index = np.empty(p)
    # Assign each variable to the group (PC) in which that variable had its maximum (absolute value) weight
    for i in range(p):
        abs_prin_comp = np.abs(prin_comp[i, :])
        group_index[i] = np.where(abs_prin_comp == np.max(abs_prin_comp))[0][0]
    group_index = np.asarray([int(element) for element in group_index]) + 1
    logger.debug('Function finished without errors')
    return group_index


def pls_group_creator(x, y):
    """
    Create groups for the columns of x based on the PC in which each variable maximizes its weight
    """
    # Obtain as much PLS components s as possible
    p = x.shape[1]
    pls = PLSRegression(n_components=np.min((x.shape[0], x.shape[1])), scale=False)
    pls.fit(x, y)
    pls_comp = pls.x_rotations_
    group_index = np.empty(p)
    # Assign each variable to the group (PC) in which that variable had its maximum (absolute value) weight
    for i in range(p):
        abs_pls_comp = np.abs(pls_comp[i, :])
        group_index[i] = np.where(abs_pls_comp == np.max(abs_pls_comp))[0][0]
    group_index = np.asarray([int(element) for element in group_index]) + 1
    logger.debug('Function finished without errors')
    return group_index


########################################################################################################################

