from __future__ import division

import numpy as np

import general_functions as gf
import graphical_code as gc
import numeric_data_generator as ngen
import additional_functions as af

"""
n=5200--p=625--sparse groups--dg_t--gamma influence 1
"""

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

# Model_selection_param
train_size = 100
validate_size = 100
error_type = 'QRE'

# Automatic simulator
n_repetitions = 50
folder = 'simulation_results/'

model_selection_param = dict(train_size=train_size, validate_size=validate_size, error_type=error_type)
data_param = dict(n=5200, num_groups=25, group_size=25, p=625, ro=0.5, num_non_zero_coef_per_group=8, num_non_zero_groups=7)
data_generator = ngen.dg_t

asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)

########################

asgl_wc_pca_d_0 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [0.0]
gl_power_weight = [1.0]
asgl_wc_pca_d_1 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [0.2]
gl_power_weight = [1.0]
asgl_wc_pca_d_2 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [0.4]
gl_power_weight = [1.0]
asgl_wc_pca_d_3 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [0.6]
gl_power_weight = [1.0]
asgl_wc_pca_d_4 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [0.8]
gl_power_weight = [1.0]
asgl_wc_pca_d_5 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [1.0]
asgl_wc_pca_d_6 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.2]
gl_power_weight = [1.0]
asgl_wc_pca_d_7 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.4]
gl_power_weight = [1.0]
asgl_wc_pca_d_8 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')
############################

l_power_weight = [1.0]
gl_power_weight = [0.0]
asgl_wc_pca_d_11 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [0.2]
asgl_wc_pca_d_12 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [0.4]
asgl_wc_pca_d_13 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [0.6]
asgl_wc_pca_d_14 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [0.8]
asgl_wc_pca_d_15 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [1.0]
asgl_wc_pca_d_16 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [1.2]
asgl_wc_pca_d_17 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

l_power_weight = [1.0]
gl_power_weight = [1.4]
asgl_wc_pca_d_18 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')
######################

model_param = [
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_0),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_1),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_2),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_3),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_4),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_5),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_6),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_7),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_8),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_11),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_12),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_13),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_14),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_15),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_16),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_17),
    dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_18)
]

results = gf.automatic_simulator(data_param=data_param, model_selection_param=model_selection_param, model_param=model_param, data_generator=data_generator, n_repetitions=n_repetitions, folder=folder)

########################################################################################################################

results = af.simulation_results_to_tables(results=results, from_file=False, table_format='row_models')

sorting = ['lasso', 'sgl', 'asgl_pca_d', 'asgl_pca_d_g1_0.0_g2_1.0', 'asgl_pca_d_g1_0.2_g2_1.0', 'asgl_pca_d_g1_0.4_g2_1.0', 'asgl_pca_d_g1_0.6_g2_1.0', 'asgl_pca_d_g1_0.8_g2_1.0', 'asgl_pca_d_g1_1.0_g2_1.0', 'asgl_pca_d_g1_1.2_g2_1.0', 'asgl_pca_d_g1_1.4_g2_1.0', 'asgl_pca_d_g1_1.0_g2_0.0', 'asgl_pca_d_g1_1.0_g2_0.2', 'asgl_pca_d_g1_1.0_g2_0.4', 'asgl_pca_d_g1_1.0_g2_0.6', 'asgl_pca_d_g1_1.0_g2_0.8', 'asgl_pca_d_g1_1.0_g2_1.0', 'asgl_pca_d_g1_1.0_g2_1.2', 'asgl_pca_d_g1_1.0_g2_1.4']
gc. boxplot_creator_by_metric_gamma_influence(results=results, interesting_metrics=['final_error'], figsize=(79, 10), sorting=sorting)

########################################################################################################################
