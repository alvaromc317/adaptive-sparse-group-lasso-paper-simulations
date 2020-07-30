from __future__ import division

import functools
import json
import multiprocessing as mp
import os

import numpy as np

import model_solvers as ms

# Load config file
root_path = os.getcwd()
config = json.loads(open(root_path + '/config').read())


# LINEAR MODEL #########################################################################################################

"""
x: Data matrix
y: response vector
intercept: boolean indicating wether to consider or not an intercept in the models
tol: inferior limit for a beta parameter to be considered as 0
tau: (only in qr models) quantile value
param: list of tuples of parameter values. Depending on the model it includes lambda, alpha, lasso weights and 
       group lasso weights
The number of cores used is the minimum between the number of cores in the PC and the number indicated in config file
"""

def lm_lasso_parallel(x, y, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.lm_lasso, x, y, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


def lm_gl_parallel(x, y, index, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.lm_gl, x, y, index, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


def lm_sgl_parallel(x, y, index, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.lm_sgl, x, y, index, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


# QUANTILE REGRESSION PREPROCESSING ####################################################################################

def qr_lasso_parallel(x, y, tau=0.5, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.qr_lasso, x, y, tau, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


def qr_gl_parallel(x, y, index, tau=0.5, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.qr_gl, x, y, index, tau, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


def qr_sgl_parallel(x, y, index, tau=0.5, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.qr_sgl, x, y, index, tau, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


def qr_al_asgl_parallel(x, y, index, tau=0.5, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.qr_al_asgl, x, y, index, tau, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


def qr_agl_asgl_parallel(x, y, index, tau=0.5, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.qr_agl_asgl, x, y, index, tau, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


def qr_asgl_parallel(x, y, index, tau=0.5, intercept=True, tol=1e-3, param=None):
    # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
    num_chunks = np.min([config['max_cores'], mp.cpu_count()])
    # We work with an index because param can be a list o values or a list of tuples depending on the penalization
    tmp_index_chunks = np.array_split(range(len(param)), num_chunks)
    # If the number of parameters is shorter than the number of threads, delete the empty groups
    index_chunks = []
    for elt in tmp_index_chunks:
        if elt.size != 0:
            index_chunks.append(elt)
    # chunks is a list with as many elements as num_chunks
    # Each element of the list is another list of tuples of parameter values
    chunks = []
    for elt in index_chunks:
        chunks.append(param[elt[0]:(1+elt[-1])])
    # Solve problem in parallel
    pool = mp.Pool(num_chunks)
    global_results = pool.map(functools.partial(ms.qr_asgl, x, y, index, tau, intercept, tol), chunks)
    pool.close()
    pool.join()
    # Re-build the output of the function
    beta_sol_list = []
    if len(param) < num_chunks:
        limit = len(param)
    else:
        limit = num_chunks
    for i in range(limit):
        beta_sol_list.extend(global_results[i])
    return beta_sol_list


########################################################################################################################
