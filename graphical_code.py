from __future__ import division

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pylab
import seaborn as sns
from matplotlib import rcParams

# Load config file
root_path = os.getcwd()
config = json.loads(open(root_path + '/config').read())

logger = logging.getLogger('graphical_code')
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


# BOXPLOT CREATION #####################################################################################################

"""
Functions associated to the results from automatic_analyzer and automatic_simulator functions
"""

def boxplot_creator_by_metric(results, interesting_metrics=('beta_error', 'final_error', 'correct_selection_rate', 'true_positive_rate', 'true_negative_rate', 'f_score'), sorting=None, destination_folder='plots/', figsize=(25, 10), labelsize = 17, plt_style='ggplot', plot_format='pdf'):
    """
    Generates a set of boxplots for each metric in interesting_metrics. Stores it in png format in the destination_folder.
    The size of the image and the size of the label text can be controlled
    """
    # Load the ggplot style and change the label text size
    plt.style.use(plt_style)
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    # Obtain model names
    stored_model_names = []
    for j in range(len(results['results'])):
        model_name = results['results'][j]['model_solver']['penalization']
        if 'asgl' in results['results'][j]['model_solver']['penalization']:
            model_name += '_' + results['results'][j]['weight_calculator']['weight_technique']
        stored_model_names.append(str(model_name))
    # For each metric considered, store the data of that metric for each model, and the name of the model
    for metric in interesting_metrics:
        list_sorting_index = []
        if sorting is None:
            list_sorting_index = range(len(stored_model_names))
        elif sorting is not None:
            for sorting_elt in sorting:
                list_sorting_index.append(np.where(np.asarray(stored_model_names)==sorting_elt)[0][0])
        labels = []
        boxplot_data = []
        for i in list_sorting_index:
            labels.append(stored_model_names[i])
            boxplot_data.append(results['results'][i]['metrics'][metric])
        # Rename al_asgl as al_sgl
        for i in range(len(labels)):
            if 'al_asgl' in labels[i]:
                labels[i] = 'al_sgl' + labels[i][7:]
        # Graphical configuration
        plt.figure(figsize=figsize)
        flierprops_style = dict(markerfacecolor='black', marker='.')
        medianprops_style = dict(color='black')
        boxprops_style = dict(color='royalblue')
        whiskerprops_style = dict(color='royalblue', linestyle="dashed")
        inside_color = ['white'] * len(boxplot_data)
        box = plt.boxplot(boxplot_data, patch_artist=True, flierprops=flierprops_style, boxprops=boxprops_style, medianprops=medianprops_style, whiskerprops=whiskerprops_style, labels=labels)
        for patch, color in zip(box['boxes'], inside_color):
            patch.set_facecolor(color)
        # Rename "final_error" as "Test error"
        title = metric
        if metric == 'final_error':
            title = 'Test error'
        if metric == 'final_error':
            title = 'Test error'
        if metric == 'correct_selection_rate':
            title = 'Correct selection rate'
        plt.title(title, fontsize=30)
        plot_name = destination_folder + metric + '.' + plot_format
        plt.savefig(plot_name, format=plot_format, dpi=500, bbox_inches='tight')
    logger.debug('Function finished without errors')


# Variation used in models where the percentaje of variance used in pca / pls  changes and this info is required to appear in the labels
def boxplot_creator_by_metric_var_pct(results, interesting_metrics=('beta_error', 'final_error', 'correct_selection_rate', 'true_positive_rate', 'true_negative_rate', 'f_score'), sorting=None, destination_folder='plots/', figsize=(25, 10), labelsize = 17, plt_style='ggplot', plot_format='pdf'):
    """
    Generates a set of boxplots for each metric in interesting_metrics. Stores it in png format in the destination_folder.
    The size of the image and the size of the label text can be controlled
    """
    # Load the ggplot style and change the label text size
    plt.style.use(plt_style)
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    # Obtain model names
    stored_model_names = []
    for j in range(len(results['results'])):
        model_name = results['results'][j]['model_solver']['penalization']
        if 'asgl' in results['results'][j]['model_solver']['penalization']:
            model_name += '_' + results['results'][j]['weight_calculator']['weight_technique']
            model_name += '_' + str(results['results'][j]['weight_calculator']['var_pct'])
        stored_model_names.append(str(model_name))
    # For each metric considered, store the data of that metric for each model, and the name of the model
    for metric in interesting_metrics:
        list_sorting_index = []
        if sorting is None:
            list_sorting_index = range(len(stored_model_names))
        elif sorting is not None:
            for sorting_elt in sorting:
                list_sorting_index.append(np.where(np.asarray(stored_model_names)==sorting_elt)[0][0])
        labels = []
        boxplot_data = []
        for i in list_sorting_index:
            labels.append(stored_model_names[i])
            boxplot_data.append(results['results'][i]['metrics'][metric])
        # Rename al_asgl as al_sgl
        for i in range(len(labels)):
            if 'al_asgl' in labels[i]:
                labels[i] = 'al_sgl' + labels[i][7:]
        # Graphical configuration
        plt.figure(figsize=figsize)
        flierprops_style = dict(markerfacecolor='black', marker='.')
        medianprops_style = dict(color='black')
        boxprops_style = dict(color='royalblue')
        whiskerprops_style = dict(color='royalblue', linestyle="dashed")
        inside_color = ['white'] * len(boxplot_data)
        box = plt.boxplot(boxplot_data, patch_artist=True, flierprops=flierprops_style, boxprops=boxprops_style, medianprops=medianprops_style, whiskerprops=whiskerprops_style, labels=labels)
        for patch, color in zip(box['boxes'], inside_color):
            patch.set_facecolor(color)
        # Rename "final_error" as "Test error"
        title = metric
        if metric == 'final_error':
            title = 'Test error'
        if metric == 'final_error':
            title = 'Test error'
        if metric == 'correct_selection_rate':
            title = 'Correct selection rate'
        plt.title(title, fontsize=30)
        plot_name = destination_folder + metric + '.' + plot_format
        plt.savefig(plot_name, format=plot_format, dpi=500, bbox_inches='tight')
    logger.debug('Function finished without errors')


# Variation used in models where the gamma changes and this info is required to appear in the labels
def boxplot_creator_by_metric_gamma_influence(results, interesting_metrics=('beta_error', 'final_error', 'correct_selection_rate', 'true_positive_rate', 'true_negative_rate', 'f_score'), sorting=None, destination_folder='plots/', figsize=(25, 10), labelsize = 17, plt_style='ggplot', plot_format='pdf'):
    """
    Generates a set of boxplots for each metric in interesting_metrics. Stores it in png format in the destination_folder.
    The size of the image and the size of the label text can be controlled
    """
    # Load the ggplot style and change the label text size
    plt.style.use(plt_style)
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    # Obtain model names
    stored_model_names = []
    for j in range(len(results['results'])):
        model_name = results['results'][j]['model_solver']['penalization']
        if 'asgl' in results['results'][j]['model_solver']['penalization']:
            model_name += '_' + results['results'][j]['weight_calculator']['weight_technique']
            if len(results['results'][j]['weight_calculator']['gl_power_weight']) == 1:
                model_name += '_g1_' + str(results['results'][j]['weight_calculator']['l_power_weight'][0])
                model_name += '_g2_' + str(results['results'][j]['weight_calculator']['gl_power_weight'][0])
        stored_model_names.append(str(model_name))
    # For each metric considered, store the data of that metric for each model, and the name of the model
    for metric in interesting_metrics:
        list_sorting_index = []
        if sorting is None:
            list_sorting_index = range(len(stored_model_names))
        elif sorting is not None:
            for sorting_elt in sorting:
                list_sorting_index.append(np.where(np.asarray(stored_model_names)==sorting_elt)[0][0])
        labels = []
        boxplot_data = []
        for i in list_sorting_index:
            labels.append(stored_model_names[i])
            boxplot_data.append(results['results'][i]['metrics'][metric])
        # Rename al_asgl as al_sgl
        for i in range(len(labels)):
            if 'al_asgl' in labels[i]:
                labels[i] = 'al_sgl' + labels[i][7:]
        # Graphical configuration
        plt.figure(figsize=figsize)
        flierprops_style = dict(markerfacecolor='black', marker='.')
        medianprops_style = dict(color='black')
        boxprops_style = dict(color='royalblue')
        whiskerprops_style = dict(color='royalblue', linestyle="dashed")
        inside_color = ['white'] * len(boxplot_data)
        box = plt.boxplot(boxplot_data, patch_artist=True, flierprops=flierprops_style, boxprops=boxprops_style, medianprops=medianprops_style, whiskerprops=whiskerprops_style, labels=labels, vert=False)
        for patch, color in zip(box['boxes'], inside_color):
            patch.set_facecolor(color)
        # Rename "final_error" as "Test error"
        title = metric
        if metric == 'final_error':
            title = 'Test error'
        if metric == 'correct_selection_rate':
            title = 'Correct selection rate'
        plt.title(title, fontsize=30)
        plot_name = destination_folder + metric + '.' + plot_format
        plt.savefig(plot_name, format=plot_format, dpi=500, bbox_inches='tight')
    logger.debug('Function finished without errors')


# INDEX RELATED PLOTS ##################################################################################################

def index_boxplot(index, destination_folder='plots/', labelsize=14, plt_style='ggplot', figsize=(6, 5), xlabel='', ylabel='', title='Genes per group', plot_format='pdf'):
    """
    Receives an index of variables (organizing them into groups)
    and generates a boxplot showing the number of variables in each group
    """
    # Define style and labelsize
    plt.style.use(plt_style)
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    # Obtain the number of variables in each group
    variables_per_group = []
    unique_index = np.unique(index)
    for idx in unique_index:
        variables_per_group.append(len(np.where(index == idx)[0]))
    variables_per_group = np.asarray(variables_per_group)
    # Graphical configuration
    plt.figure(figsize=figsize)
    flierprops_style = dict(markerfacecolor='black', marker='.')
    medianprops_style = dict(color='black')
    boxprops_style = dict(color='royalblue')
    whiskerprops_style = dict(color='royalblue', linestyle="dashed")
    inside_color = ['white']
    box = plt.boxplot(variables_per_group, patch_artist=True, flierprops=flierprops_style, boxprops=boxprops_style, medianprops=medianprops_style, whiskerprops=whiskerprops_style, labels=[''])
    for patch, color in zip(box['boxes'], inside_color):
        patch.set_facecolor(color)
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plot_name = destination_folder + 'index_boxplot' + '.' + plot_format
    plt.savefig(plot_name, format=plot_format, dpi=500, bbox_inches='tight')


def index_barplot(index, destination_folder='plots/', labelsize=14, plt_style='ggplot', figsize=(8, 5), xlabel='', ylabel='', title='', plot_format='pdf'):
    """
    Receives an index of variables (organizing them into groups)
    and generates a barplot showing the number of variables in each group
    """
    plt.style.use(plt_style)
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    variables_per_group = []
    unique_index = np.unique(index)
    for idx in unique_index:
        variables_per_group.append(len(np.where(index == idx)[0]))
    variables_per_group = np.asarray(variables_per_group)
    plt.figure(figsize=figsize)
    plt.bar(unique_index, variables_per_group, color='royalblue')
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plot_name = destination_folder + 'index_barplot' + '.' + plot_format
    plt.savefig(plot_name, format=plot_format, dpi=500, bbox_inches='tight')


# SIGNIFICANCE LEVEL OF VARIABLES ######################################################################################

def variables_probability_heatmap(results, sorting=None, destination_folder='plots/', labelsize=20, figsize=(25, 13), plt_style='ggplot', xlabel='Significant variables', ylabel='', title='Probability of being a significant variable', plot_format='png'):
    """
    Receives a results file with the indexed significant variables, and generates a heat-map of being a significant variable
    based on the number of times a variable has been chosen as significant along many iterations of the models
    """
    plt.style.use(plt_style)
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    # Obtain a list of all the significant variables in any of the iterations, in any of the models computed.
    # Obtain the list with the names of the models solved.
    global_significant_variables = []
    stored_model_names = []
    for j in range(len(results['results'])):
        model_name = results['results'][j]['model_solver']['penalization']
        if 'asgl' in results['results'][j]['model_solver']['penalization']:
            model_name += '_' + results['results'][j]['weight_calculator']['weight_technique']
        stored_model_names.append(str(model_name))
        for index_model in results['results'][j]['extra_metrics']['index_non_zero_pred_beta']:
            global_significant_variables = global_significant_variables + index_model
    global_significant_variables = np.unique(global_significant_variables)
    # Sort the models based on sorting parameter
    list_sorting_index = []
    if sorting is None:
        list_sorting_index = range(len(stored_model_names))
    elif sorting is not None:
        for sorting_elt in sorting:
            list_sorting_index.append(np.where(np.asarray(stored_model_names) == sorting_elt)[0][0])
    n_models = len(results['results'])
    n_variables = len(global_significant_variables)
    probability_of_significance = np.zeros((n_models, n_variables))
    # Obtain for each globally significant variable, the number of times that variable was chosen in each model
    for i in range(len(list_sorting_index)):
        # i is the index in which variables must be displayed
        # j is the index in which the models are sorted
        j = list_sorting_index[i]
        for index_significant_variables in results['results'][j]['extra_metrics']['index_non_zero_pred_beta']:
            for variable_index in index_significant_variables:
                pos = np.where(global_significant_variables == variable_index)[0]
                probability_of_significance[i, pos] += 1
        probability_of_significance[i,:] = probability_of_significance[i,:] / len(results['results'][j]['extra_metrics']['index_non_zero_pred_beta'])
    # If sorted is defined then the probability matrix is sorted based on the values of the first model
    if sorting is not None:
        sorting_variables_index = np.argsort(-probability_of_significance[0,:])
        global_significant_variables = global_significant_variables[sorting_variables_index]
        for i in range(n_models):
            probability_of_significance[i, :] = probability_of_significance[i, sorting_variables_index]
    sorted_model_names = np.asarray(stored_model_names)[list_sorting_index].tolist()
    plt.figure(figsize=figsize)
    sns.heatmap(probability_of_significance, linewidth=1e-15, yticklabels=sorted_model_names, xticklabels='', cmap="Blues")
    plt.show()
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)
    plot_name = destination_folder + 'probability_heatmap' + '.' + plot_format
    plt.savefig(plot_name, format=plot_format, dpi=500, bbox_inches='tight')
    results = dict(global_significance_variables=global_significant_variables, probability_of_significance=probability_of_significance, model_names=sorted_model_names)
    return results


########################################################################################################################
