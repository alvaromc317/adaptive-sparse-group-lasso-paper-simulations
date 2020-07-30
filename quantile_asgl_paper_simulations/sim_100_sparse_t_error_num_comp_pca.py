from __future__ import division

import numpy as np

import general_functions as gf
import graphical_code as gc
import numeric_data_generator as ngen
import additional_functions as af

"""
n=5200--p=100--sparse groups--dg_t--num comp influence--pca
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
train_size = 500
validate_size = 500
error_type = 'QRE'

# Automatic simulator
n_repetitions = 50
folder = 'simulation_results/'

model_selection_param = dict(train_size=train_size, validate_size=validate_size, error_type=error_type)
data_param = dict(n=6000, num_groups=10, group_size=10, p=100, ro=0.5, num_non_zero_coef_per_group=6, num_non_zero_groups=5)
data_generator = ngen.dg_t

asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)

var_pct = 0.1
asgl_wc_pca_d_1 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.2
asgl_wc_pca_d_2 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.3
asgl_wc_pca_d_3 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.4
asgl_wc_pca_d_4 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.5
asgl_wc_pca_d_5 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.6
asgl_wc_pca_d_6 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.7
asgl_wc_pca_d_7 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.8
asgl_wc_pca_d_8 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 0.9
asgl_wc_pca_d_9 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

var_pct = 1
asgl_wc_pca_d_10 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

model_param = [dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_1),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_2),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_3),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_4),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_5),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_6),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_7),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_8),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_9),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d_10)
]

results = gf.automatic_simulator(data_param=data_param, model_selection_param=model_selection_param, model_param=model_param, data_generator=data_generator, n_repetitions=n_repetitions, folder=folder)

########################################################################################################################

results = af.simulation_results_to_tables(results=results, from_file=False, table_format='row_models')

sorting = ['lasso', 'sgl', 'asgl_pca_d_0.1', 'asgl_pca_d_0.2', 'asgl_pca_d_0.3', 'asgl_pca_d_0.4', 'asgl_pca_d_0.5', 'asgl_pca_d_0.6', 'asgl_pca_d_0.7', 'asgl_pca_d_0.8', 'asgl_pca_d_0.9', 'asgl_pca_d_1']
gc. boxplot_creator_by_metric_var_pct(results=results, interesting_metrics=['final_error'], figsize=(33, 10), sorting=sorting)

########################################################################################################################
