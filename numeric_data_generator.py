from __future__ import division

import json
import logging
import os

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import cauchy

# Load config file
root_path = os.getcwd()
config = json.loads(open(root_path + '/config').read())

logger = logging.getLogger('numeric_dg')
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


########################################################################################################################


def dg_enet(n=100, p=40, noise_group=True):
    """
    Regularization_and_variable_selection_via_elastic_net__hastie__zou__2015
    Example 4
    """
    if noise_group:
        group_index = np.asarray([1]*5 + [2]*5 + [3]*5 + [4]*(p-15))
    else:
        noise =range(16, (p+1))
        group_index = np.asarray([1] * 5 + [2] * 5 + [3] * 5 + noise)
    beta = np.r_[np.repeat(3, 15), np.zeros(p-15)]
    z1_tmp = np.random.normal(0, 1, n)
    z2_tmp = np.random.normal(0, 1, n)
    z3_tmp = np.random.normal(0, 1, n)
    z1 = np.c_[z1_tmp, z1_tmp, z1_tmp, z1_tmp, z1_tmp]
    z2 = np.c_[z2_tmp, z2_tmp, z2_tmp, z2_tmp, z2_tmp]
    z3 = np.c_[z3_tmp, z3_tmp, z3_tmp, z3_tmp, z3_tmp]
    x1 = z1 + np.random.multivariate_normal(np.repeat(0, 5), np.diag([0.05]*5), n)
    x2 = z2 + np.random.multivariate_normal(np.repeat(0, 5), np.diag([0.05] * 5), n)
    x3 = z3 + np.random.multivariate_normal(np.repeat(0, 5), np.diag([0.05] * 5), n)
    x4 = np.random.multivariate_normal(np.repeat(0, (p-15)), np.diag([1] * (p-15)), n)
    x = np.c_[x1, x2, x3, x4]
    e = np.random.normal(0, 15, n)
    y = np.dot(x, beta) + e
    data = dict(x=x, y=y, true_beta=beta, index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg_t(n=150, num_groups=40, group_size=40, p=1600, ro=0.5, num_non_zero_coef_per_group=8, num_non_zero_groups=7):
    """
    Generate a dataset formed by groups of variables. All the groups have the same size.
    num_non_zero_coef_per_group
    num_non_zero_groups
    :param n: number of observations
    :param num_groups: number of groups
    :param group_size: size of the groups
    :param p: number of variables
    :param ro: correlation between variables inside a group
    :param num_non_zero_coef_per_group: controls how many significant variables we want
    :param num_non_zero_groups: controls how many groups we want to store significant variables
    :return: x, y index
    If num_non_zero_coef_per_group=8 and num_non_zero_groups=7 we will have 8 groups with 7 significant variables,
    and num_groups-num_non_zero_coef_per_group groups with all non-significant variables
    The error distribution is a student t with 3 degrees of freedom
    """
    p = num_groups*group_size
    group_levels = np.arange(1, (num_groups+1), 1)
    num_non_zeros_per_group = [0]*num_groups
    num_non_zeros_per_group[0:num_non_zero_groups] = [num_non_zero_coef_per_group]*num_non_zero_groups
    group_index = np.repeat(group_levels, [group_size] * num_groups, axis=0)
    s = np.zeros((p, p))
    for level in group_levels:
        s[(level - 1) * group_size:level * group_size, (level - 1) * group_size:level * group_size] = ro
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    betas = np.zeros(p)
    for i in range(num_groups):
        betas[(i*group_size):((i+1)*group_size)] = np.arange(1, group_size+1, 1)
    for i in range(num_groups):
        betas[((group_levels[i] - 1) * group_size + num_non_zeros_per_group[i]):group_levels[i] * group_size] = 0
    e = np.random.standard_t(3, size=n)
    y = np.dot(x, betas) + e
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg_cauchy(n=150, num_groups=40, group_size=40, p=1600, ro=0.5, num_non_zero_coef_per_group=8, num_non_zero_groups=7, loc=0, scale=3):
    """
    Generate a dataset formed by groups of variables. All the groups have the same size.
    num_non_zero_coef_per_group
    num_non_zero_groups
    :param n: number of observations
    :param num_groups: number of groups
    :param group_size: size of the groups
    :param p: number of variables
    :param ro: correlation between variables inside a group
    :param num_non_zero_coef_per_group: controls how many significant variables we want
    :param num_non_zero_groups: controls how many groups we want to store significant variables
    :param loc: location parameter for cauchy error
    :param scale: scale parameter for cauchy error
    :return: x, y index
    If num_non_zero_coef_per_group=8 and num_non_zero_groups=7 we will have 8 groups with 7 significant variables,
    and num_groups-num_non_zero_coef_per_group groups with all non-significant variables
    The error distribution is a student t with 3 degrees of freedom
    """
    p = num_groups*group_size
    group_levels = np.arange(1, (num_groups+1), 1)
    num_non_zeros_per_group = [0]*num_groups
    num_non_zeros_per_group[0:num_non_zero_groups] = [num_non_zero_coef_per_group]*num_non_zero_groups
    group_index = np.repeat(group_levels, [group_size] * num_groups, axis=0)
    s = np.zeros((p, p))
    for level in group_levels:
        s[(level - 1) * group_size:level * group_size, (level - 1) * group_size:level * group_size] = ro
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    betas = np.zeros(p)
    for i in range(num_groups):
        betas[(i*group_size):((i+1)*group_size)] = np.arange(1, group_size+1, 1)
    for i in range(num_groups):
        betas[((group_levels[i] - 1) * group_size + num_non_zeros_per_group[i]):group_levels[i] * group_size] = 0
    e  = cauchy.rvs(loc=loc, scale=scale, size=n)
    y = np.dot(x, betas) + e
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg_chisq(n=150, num_groups=40, group_size=40, p=1600, ro=0.5, num_non_zero_coef_per_group=8, num_non_zero_groups=7, df=3):
    """
    Generate a dataset formed by groups of variables. All the groups have the same size.
    num_non_zero_coef_per_group
    num_non_zero_groups
    :param n: number of observations
    :param num_groups: number of groups
    :param group_size: size of the groups
    :param p: number of variables
    :param ro: correlation between variables inside a group
    :param num_non_zero_coef_per_group: controls how many significant variables we want
    :param num_non_zero_groups: controls how many groups we want to store significant variables
    :param df: degrees of freedom in the chisq distribution
    :return: x, y index
    If num_non_zero_coef_per_group=8 and num_non_zero_groups=7 we will have 8 groups with 7 significant variables,
    and num_groups-num_non_zero_coef_per_group groups with all non-significant variables
    The error distribution is a student t with 3 degrees of freedom
    """
    p = num_groups*group_size
    group_levels = np.arange(1, (num_groups+1), 1)
    num_non_zeros_per_group = [0]*num_groups
    num_non_zeros_per_group[0:num_non_zero_groups] = [num_non_zero_coef_per_group]*num_non_zero_groups
    group_index = np.repeat(group_levels, [group_size] * num_groups, axis=0)
    s = np.zeros((p, p))
    for level in group_levels:
        s[(level - 1) * group_size:level * group_size, (level - 1) * group_size:level * group_size] = ro
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    betas = np.zeros(p)
    for i in range(num_groups):
        betas[(i*group_size):((i+1)*group_size)] = np.arange(1, group_size+1, 1)
    for i in range(num_groups):
        betas[((group_levels[i] - 1) * group_size + num_non_zeros_per_group[i]):group_levels[i] * group_size] = 0
    e = np.random.chisquare(df, size=n)
    y = np.dot(x, betas) + e
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg_stn_varying_group_size(n=150, p=750, ro=0.5, stn=3, group_sizes_general=(5, 15, 30), num_groups_general=(15, 15, 15), num_non_zero_coef=(3, 6, 10), num_non_zero_groups=(3, 3, 3)):
    """
    Generate a dataset formed by groups of variables. The groups have different sizes
    :param n: number of observations
    :param p: number of variables
    :param ro: correlation between variables inside each group
    :param stn: The signal-to-noise ratio
    :param group_sizes_general: The size of each type of gropu that we want to generate
    :param num_groups_general: How many groups of each size we want to generate
    :param num_non_zero_coef: How many significant variables we want inside each type of group
    :param num_non_zero_groups: How many groups with significant variables we want
    :return: x, y, index
    Using the predefined values, we would have 5 groups of size 15, including 3 groups with 3 significant variables
    and 12 groups of noise
    """
    total_num_groups = int(np.sum(num_groups_general))
    group_levels = np.arange(1, total_num_groups+1, 1)
    group_sizes = []
    betas = []
    for i in range(len(group_sizes_general)):
        group_sizes = group_sizes + [group_sizes_general[i]]*num_groups_general[i]
        tmp_betas = range(1, int(num_non_zero_coef[i])+1) + [0]*(group_sizes_general[i]-num_non_zero_coef[i])
        tmp_betas2 = [0] * group_sizes_general[i]
        betas = betas + tmp_betas * num_non_zero_groups[i] + tmp_betas2 * (num_groups_general[i] - num_non_zero_groups[i])
    betas = np.asarray(betas)
    group_index = np.repeat(group_levels, group_sizes)
    s = np.zeros((p, p))
    inf_lim = 0
    sup_lim = 0
    for i in range(len(group_levels)):
        sup_lim = sup_lim + group_sizes[i]
        s[inf_lim:sup_lim, inf_lim:sup_lim] = ro
        inf_lim = sup_lim
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    std_error = np.sqrt(np.dot(np.dot(betas.T, s), betas)/stn)
    e = np.random.normal(0, std_error, n)
    y = np.dot(x, betas) + e
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg(n, p, ro, sigma, error_type, density, shape):
    """
    Dataset generation based on 1996 Tibshirani and 2004 Friedmann
    """
    # Error distribution
    e = None
    if error_type == 'N':
        e = np.random.normal(0, 1, n)
    elif error_type == 'DE':
        e = np.random.laplace(0, 1, n)
    elif error_type == 'M':
        e = 0.1*np.random.normal(0, 5, n) + 0.9*np.random.normal(0, 1, n)
    s = np.zeros((p, p))
    k_coefficient = np.linspace(1, p, p)
    for i in range(1, p+1):
        s[i-1, ] = ro**(np.abs(i - k_coefficient))
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    # Density level
    beta = None
    if shape == 'small':
        if density == 'dense':
            beta = np.repeat(0.85, p)
        elif density == 'sparse':
            beta = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
        elif density == 'v_sparse':
            beta = np.array([5, 0, 0, 0, 0, 0, 0, 0])
    elif shape == 'large':
        if density == 'dense':
            beta = np.random.normal(0, 1, p)
        elif density == 'sparse':
            beta = np.concatenate((np.random.normal(0, 1, 30), np.repeat(0, 270)), axis=0)
        elif density == 'v_sparse':
            beta = np.concatenate((np.random.normal(0, 1, 3), np.repeat(0, 297)), axis=0)
        signal_standardization = np.linalg.norm(np.dot(sqrtm(s), beta), ord=2)
        beta = beta / signal_standardization
        sigma = np.sqrt(sigma / np.var(e))
    error2 = sigma*e
    y = np.dot(x, beta) + error2
    data = dict(x=x, y=y, true_beta=beta)
    logger.debug('Function finished without errors')
    return data


def dg_hierarchical(n=200, p=100, ro=0.2, group_size=10, error_type='CHI4', beta_type='int'):
    """
    Sparse_group_variable_selection_based_on_quantile_hierarchical_lasso__zhao__zhang__liu__2014
    Example 2
    """
    e = None
    if error_type == 'N1':
        e = np.random.normal(0, 1, n)
    elif error_type == 'N4':
        e = np.random.normal(0, 4, n)
    elif error_type == 'DE':
        e = np.random.laplace(0, 1, n)
    elif error_type == 'M':
        e = 0.1*np.random.normal(0, 5, n) + 0.9*np.random.normal(0, 1, n)
    elif error_type == 'CHI4':
        e = np.random.chisquare(df=4, size=n)
    elif error_type == 'CHI2':
        e = np.random.chisquare(df=2, size=n)
    group_levels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    num_non_zeros_per_group = np.array([10, 8, 6, 4, 2, 1, 0, 0, 0, 0])*int((group_size/10))
    group_index = np.repeat(group_levels, [group_size]*10, axis=0)
    s = np.zeros((p, p))
    for level in group_levels:
        s[(level - 1)*group_size:level*group_size, (level - 1)*group_size:level*group_size] = ro
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    betas = None
    if beta_type == 'int':
        betas = np.random.choice([-1, 1], size=p, replace=True)
    elif beta_type == 'N':
        betas = np.random.normal(0, 1, p)
    for i in range(len(group_levels)):
        betas[((group_levels[i] - 1)*group_size + num_non_zeros_per_group[i]):group_levels[i]*group_size] = 0
    y = np.dot(x, betas) + e
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg_stn_10groups(n=150, p=400, ro=0.5, group_size=40, n_non_zero_group=2, stn=3, error_type='N'):
    """
    Generates 10 groups of size group_size with 5 significant variables in n_non_zero_group groupd
    param stn: controlls the signal to noise ratio
    """
    e = None
    group_levels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    num_non_zeros_per_group = [0]*10
    num_non_zeros_per_group[0:n_non_zero_group] = [5]*n_non_zero_group
    group_index = np.repeat(group_levels, [group_size] * 10, axis=0)
    s = np.zeros((p, p))
    for level in group_levels:
        s[(level - 1) * group_size:level * group_size, (level - 1) * group_size:level * group_size] = ro
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    betas = np.zeros(p)
    for i in range(len(group_levels)):
        betas[(i*group_size):((i+1)*group_size)] = np.arange(1, group_size+1, 1)
    for i in range(len(group_levels)):
        betas[((group_levels[i] - 1) * group_size + num_non_zeros_per_group[i]):group_levels[i] * group_size] = 0
    std_error = np.sqrt(np.dot(np.dot(betas.T, s), betas)/stn)
    if error_type == 'N':
        e = np.random.normal(0, std_error, n)
    y = np.dot(x, betas) + e
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg_stn_20groups(n=150, p=40, ro=0.5, group_size=2, n_non_zero_group=3, stn=3, error_type='N'):
    """
    Generates 20 groups of size group_size with 5 significant variables in n_non_zero_group groupd
    param stn: controlls the signal to noise ratio
    """
    e = None
    group_levels = np.arange(1, 21, 1)
    num_non_zeros_per_group = [0]*20
    num_non_zeros_per_group[0:n_non_zero_group] = [5]*n_non_zero_group
    group_index = np.repeat(group_levels, [group_size] * 20, axis=0)
    s = np.zeros((p, p))
    for level in group_levels:
        s[(level - 1) * group_size:level * group_size, (level - 1) * group_size:level * group_size] = ro
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, p), s, n)
    betas = np.zeros(p)
    for i in range(len(group_levels)):
        betas[(i*group_size):((i+1)*group_size)] = np.arange(1, group_size+1, 1)
    for i in range(len(group_levels)):
        betas[((group_levels[i] - 1) * group_size + num_non_zeros_per_group[i]):group_levels[i] * group_size] = 0
    std_error = np.sqrt(np.dot(np.dot(betas.T, s), betas)/stn)
    if error_type == 'N':
        e = np.random.normal(0, std_error, n)
    y = np.dot(x, betas) + e
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data


########################################################################################################################

def stn_calculator(data_generator, data_param):
    data = data_generator(**data_param)
    x = data['x']
    y = data['y']
    true_beta = data['true_beta']
    error = y - np.dot(x, true_beta)
    s = np.cov(x.T)
    var_error = np.var(error)
    stn = np.dot(np.dot(true_beta.T, s), true_beta)/var_error
    return stn


