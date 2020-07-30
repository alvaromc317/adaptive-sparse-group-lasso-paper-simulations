from __future__ import division

import csv

import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import scale

import additional_functions as af
import general_functions as gf
import graphical_code as gc

"""
Code for the analysis of the scheetz genetic dataset formed by 120 rats and 31000 genes
"""

# LOAD THE DATASET #####################################################################################################

"""
The dataset was originally in log-scale, but the data loaded here is in the original scale. 
"""

x = np.loadtxt("data/x.txt", delimiter="\t", skiprows=1)
y = np.loadtxt("data/y.txt", delimiter="\t", skiprows=1)

with open("data/x_names.csv") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    x_names = np.asarray([r[0][1:] for r in reader])

# SELECT THESE WITH LARGE CORRELATION
cor = np.zeros(x.shape[1])
for i in range(len(cor)):
    cor[i] = pearsonr(x[:,i], y)[0]

min_cor = 0.5
large_cor_index = np.where(np.abs(cor)>min_cor)[0]
x = x[:,large_cor_index]
x_names = x_names[large_cor_index]

x = scale(x)
y = scale(y)

# OBTAIN A GROUP INDEX FOR THE GENES
index = af.pca_group_creator(x)

gc.index_boxplot(index)

########################################################################################################################

"""
Analysis using function automatic_analyzer and solving an asgl_pct and a pls_pct model
"""

model = 'qr'
intercept = True
tol = 1e-4
parallel = True
lambda1 = (10.0**np.arange(-3, 1.01, 0.2)).tolist()
alpha = (np.r_[np.arange(0.0, 0.3, 0.03), np.arange(0.3, 0.7, 0.1), np.arange(0.7, 0.99, 0.03), np.array([1])]).tolist()

tau = 0.5

l_power_weight = np.arange(0, 1.21, 0.2).tolist()
gl_power_weight = np.arange(0, 1.21, 0.2).tolist()
var_pct = 0.8

train_size = 80
validate_size = 20
error_type = 'QRE'

# Automatic simulator
n_repetitions = 1
folder = 'simulation_results/'

model_selection_param = dict(train_size=train_size, validate_size=validate_size, error_type=error_type)
asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)
asgl_wc_pca_d = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pca_d')
asgl_wc_pls_d = dict(l_power_weight=l_power_weight, gl_power_weight=gl_power_weight, var_pct=var_pct, weight_technique='pls_d')

model_param = [dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pls_d)]

dataset= dict(x=x, y=y, index=index)

results = gf.automatic_analyzer(dataset=dataset, model_selection_param=model_selection_param, model_param=model_param, n_repetitions=n_repetitions, folder=folder)

########################################################################################################################

tau = 0.3
asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)
model_param = [dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pls_d)]
results3 = gf.automatic_analyzer(dataset=dataset, model_selection_param=model_selection_param, model_param=model_param, n_repetitions=n_repetitions, folder=folder)

########################################################################################################################

tau = 0.7
asgl_model = dict(model='qr', penalization='asgl', intercept=intercept, tol=tol, lambda1=lambda1, alpha=alpha, tau=tau, parallel=parallel)
model_param = [dict(model_solver=asgl_model, weight_calculator=asgl_wc_pca_d),
               dict(model_solver=asgl_model, weight_calculator=asgl_wc_pls_d)]
results7 = gf.automatic_analyzer(dataset=dataset, model_selection_param=model_selection_param, model_param=model_param, n_repetitions=n_repetitions, folder=folder)

########################################################################################################################

"""
Processing the results obtained
"""

gc. boxplot_creator_by_metric(results=results, interesting_metrics=['final_error', 'non_zero_pred_beta'], figsize=(25, 10), sorting = ['lasso', 'sgl', 'asgl_pca_d', 'asgl_pls_d'])
significance = gc. variables_probability_heatmap(results=results, sorting=['asgl_pca_d', 'asgl_pls_d', 'sgl', 'lasso'])

model_names = ['lasso', 'sgl', 'asgl_pca_d', 'asgl_pls_d']
probability_of_significance = significance['probability_of_significance']

# Number of genes above a threshold
threshold = 0.5
for i in range(len(model_names)):
    tmp_prob = probability_of_significance[i , :]
    num_genes = len(np.where(tmp_prob>=threshold)[0])
    print("Model: {}. Threshold: {}. Number of genes: {}".format(model_names[i], threshold, num_genes))
    print(np.sort(-1*tmp_prob)[0:30]*(-1))

########################################################################################################################

"""
Analysis of different quantiles
"""

def probability_of_significance_calculator(list_results, model='asgl_pls_d', n_var=3734):
    response_models = []
    i = 0
    general_probability_of_significance = np.zeros((len(list_results), n_var))
    for results in list_results:
        stored_model_names = []
        for j in range(len(results['results'])):
            model_name = results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                model_name += '_' + results['results'][j]['weight_calculator']['weight_technique']
            stored_model_names.append(str(model_name))
        model_index = np.where(np.asarray(stored_model_names)==model)[0][0]
        response_models.append(stored_model_names[model_index]+'_tau'+str(results['results'][0]['model_solver']['tau']))
        for index_significant_variables in results['results'][model_index]['extra_metrics']['index_non_zero_pred_beta']:
            general_probability_of_significance[i, index_significant_variables] += 1.0
        general_probability_of_significance[i,:] = general_probability_of_significance[i,:] / len(results['results'][model_index]['extra_metrics']['index_non_zero_pred_beta'])
        i += 1
    response = dict(models=response_models, probability_of_significance=general_probability_of_significance)
    return response

list_results = (results, results3, results7)

# Collect the probabilities of each model for different taus
lasso_significance = probability_of_significance_calculator(list_results, model='lasso', n_var=3734)
sgl_significance = probability_of_significance_calculator(list_results, model='sgl', n_var=3734)
asgl_pca_significance = probability_of_significance_calculator(list_results, model='asgl_pca_d', n_var=3734)
asgl_pls_significance = probability_of_significance_calculator(list_results, model='asgl_pls_d', n_var=3734)

model_significance = [dict(model='lasso', significance_dict=lasso_significance),
                      dict(model='sgl', significance_dict=sgl_significance),
                      dict(model='asgl_pca_d', significance_dict=asgl_pca_significance),
                      dict(model='asgl_pls_d', significance_dict=asgl_pls_significance)]

threshold = 0.5
intersection = dict()
for elt in model_significance:
    tmp_results = []
    p = elt['significance_dict']['probability_of_significance']
    n_models = p.shape[0]
    for j in range(p.shape[1]):
        if np.sum(p[:, j] > threshold) == n_models:
            tmp_results.append(j)
    intersection[elt['model']]=tmp_results

########################################################################################################################



