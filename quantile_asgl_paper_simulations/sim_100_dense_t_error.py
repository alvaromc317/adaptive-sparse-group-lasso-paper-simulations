from __future__ import division

import numpy as np

import general_functions as gf
import graphical_code as gc
import numeric_data_generator as ngen
import additional_functions as af

"""
n=5400--p=100--dense groups--dg_t
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
train_size = 200
validate_size = 200
error_type = 'QRE'

# Automatic simulator
n_repetitions = 100
folder = ''

model_selection_param = dict(train_size=train_size, validate_size=validate_size, error_type=error_type)
data_param = dict(n=5200, num_groups=10, group_size=10, p=100, ro=0.5, num_non_zero_coef_per_group=10, num_non_zero_groups=3)
data_generator = ngen.dg_t

asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)

asgl_wc_pca_d = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')
asgl_wc_pls_d = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pls_d')
asgl_wc_unpenalized_qr = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, weight_technique='unpenalized_qr')

model_param = [dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pls_d),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_unpenalized_qr)]


results = gf.automatic_simulator(data_param=data_param, model_selection_param=model_selection_param, model_param=model_param, data_generator=data_generator, n_repetitions=n_repetitions, folder=folder)

########################################################################################################################

results = af.simulation_results_to_tables(results=results, from_file=False, table_format='row_models')

sorting = ['lasso', 'sgl', 'asgl_pca_d','asgl_pls_d', 'asgl_unpenalized']
gc. boxplot_creator_by_metric(results=results, interesting_metrics=['final_error', 'non_zero_pred_beta'], figsize=(25, 10), sorting=sorting)

########################################################################################################################
