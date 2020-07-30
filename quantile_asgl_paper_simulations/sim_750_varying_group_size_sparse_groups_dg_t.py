from __future__ import division

import numpy as np

import general_functions as gf
import graphical_code as gc
import numeric_data_generator as ngen
import additional_functions as af

"""
n=5200--p=750--varying group size sparse groups--dg_t
"""

# Model solver
model = 'qr'
penalization = 'asgl'
intercept = True
tol = 1e-4
parallel = True
lambda1 = (10.0**np.arange(-3, 0.61, 0.2)).tolist()
alpha = (np.r_[np.arange(0.01, 0.4, 0.02), np.arange(0.4, 0.8, 0.1), np.arange(0.8, 1, 0.02)]).tolist()
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
n_repetitions = 20
folder = 'simulation_results/'

model_selection_param = dict(train_size=train_size, validate_size=validate_size, error_type=error_type)

data_param = dict(n=5200, p=750, ro=0.5, stn=3, group_sizes_general=[5, 15, 30], num_groups_general=[15, 15, 15], num_non_zero_coef=[3, 6, 10], num_non_zero_groups=[3, 3, 3])
data_generator = ngen.dg_stn_varying_group_size

l_model = dict(model='qr', penalization='lasso', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)
sgl_model = dict(model='qr', penalization='sgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)
al_asgl_model = dict(model='qr', penalization='al_asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)
asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)

wc = None
al_asgl_wc_pca_1 = dict(l_power_weight=l_power_weight, weight_technique='pca_1')
asgl_wc_pca_1 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, weight_technique='pca_1')
al_asgl_wc_pca_d = dict(l_power_weight=l_power_weight, weight_technique='pca_d')
asgl_wc_pca_d = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')

al_asgl_wc_pls_1 = dict(l_power_weight=l_power_weight, weight_technique='pls_1')
asgl_wc_pls_1 = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, weight_technique='pls_1')
al_asgl_wc_pls_d = dict(l_power_weight=l_power_weight, weight_technique='pls_d')
asgl_wc_pls_d = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pls_d')

model_param = [dict(model_solver=l_model, weight_calculator=wc),
               dict(model_solver=sgl_model, weight_calculator=wc),
               dict(model_solver=al_asgl_model, weight_calculator=al_asgl_wc_pca_1),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_1),
               dict(model_solver=al_asgl_model, weight_calculator=al_asgl_wc_pca_d),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d),
               dict(model_solver=al_asgl_model, weight_calculator=al_asgl_wc_pls_1),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pls_1),
               dict(model_solver=al_asgl_model, weight_calculator=al_asgl_wc_pls_d),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pls_d)]

results = gf.automatic_simulator(data_param=data_param, model_selection_param=model_selection_param, model_param=model_param, data_generator=data_generator, n_repetitions=n_repetitions, folder=folder)

########################################################################################################################

results = af.simulation_results_to_tables(results=results, from_file=False, table_format='row_models')

sorting = ['lasso', 'sgl', 'asgl_pca_d','asgl_pls_d', 'asgl_unpenalized']
gc. boxplot_creator_by_metric(results=results, interesting_metrics=['final_error', 'non_zero_pred_beta'], figsize=(25, 10), sorting=sorting)

########################################################################################################################
