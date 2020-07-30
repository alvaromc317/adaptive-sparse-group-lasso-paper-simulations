from __future__ import division

import json
import logging
import os

import numpy as np
from tabulate import tabulate

# Load config file
root_path = os.getcwd()
config = json.loads(open(root_path + '/config').read())

logger = logging.getLogger('json_treatment')
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

def simulation_results_to_tables(results, from_file=True, table_format='row_models'):
    """
    Transforms the json from the automatic simulator into tables of results
    """
    if from_file is True:
        with open(results) as json_file:
            results = json.load(json_file)
    if table_format == 'row_metrics_one':
        for j in range(len(results['results'])):
            header_name = 'Model: ' + results['results'][j]['model_solver']['model'] + ', Penalization: ' + results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                header_name += ' ' + results['results'][j]['weight_calculator']['weight_technique']
            headers = [header_name, 'Value: [media, sd]']
            sorted_summary = sorted([(k, v) for k, v in results['results'][j]['summary'].items()])
            print(tabulate(sorted_summary, headers=headers))
    elif table_format == 'row_metrics_all':
        tmp_results = []
        headers = ['Metrics']
        for j in range(len(results['results'])):
            tmp_results.append(results['results'][j]['summary'])
            header_name = 'Penalization: ' + results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                header_name += ' ' + results['results'][j]['weight_calculator']['weight_technique']
            headers.append(header_name)
        one_table_result = {}
        # noinspection PyCompatibility
        for k in tmp_results[0].iterkeys():
            one_table_result[k] = tuple(d[k] for d in tmp_results)
        print(tabulate([(k,) + v for k, v in one_table_result.items()], headers=headers))
    elif table_format == 'row_models':
        table = []
        header = None
        for j in range(len(results['results'])):
            s = results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                s = s + '-' + results['results'][j]['weight_calculator']['weight_technique']
            header = ['Model']
            model_table = [s]
            for key in results['results'][j]['metrics'].keys():
                model_table.append(results['results'][j]['summary'][key])
                header.append(key)
            table.append(model_table)
        print(tabulate(table, headers=header))
    return results


def simulation_results_to_tables_var_pct(results, from_file=True, table_format='row_models'):
    """
    Transforms the json from the automatic simulator into tables of results
    """
    if from_file is True:
        with open(results) as json_file:
            results = json.load(json_file)
    if table_format == 'row_metrics_one':
        for j in range(len(results['results'])):
            header_name = 'Model: ' + results['results'][j]['model_solver']['model'] + ', Penalization: ' + results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                header_name += ' ' + results['results'][j]['weight_calculator']['weight_technique']
            headers = [header_name, 'Value: [media, sd]']
            sorted_summary = sorted([(k, v) for k, v in results['results'][j]['summary'].items()])
            print(tabulate(sorted_summary, headers=headers))
    elif table_format == 'row_metrics_all':
        tmp_results = []
        headers = ['Metrics']
        for j in range(len(results['results'])):
            tmp_results.append(results['results'][j]['summary'])
            header_name = 'Penalization: ' + results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                header_name += ' ' + results['results'][j]['weight_calculator']['weight_technique']
            headers.append(header_name)
        one_table_result = {}
        # noinspection PyCompatibility
        for k in tmp_results[0].iterkeys():
            one_table_result[k] = tuple(d[k] for d in tmp_results)
        print(tabulate([(k,) + v for k, v in one_table_result.items()], headers=headers))
    elif table_format == 'row_models':
        table = []
        header = None
        for j in range(len(results['results'])):
            s = results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                s = s + '-' + results['results'][j]['weight_calculator']['weight_technique'] + '-' + str(results['results'][j]['weight_calculator']['var_pct'])
            header = ['Model']
            model_table = [s]
            for key in results['results'][j]['metrics'].keys():
                model_table.append(results['results'][j]['summary'][key])
                header.append(key)
            table.append(model_table)
        print(tabulate(table, headers=header))
    return results


def simulation_results_to_tables_spca(results, from_file=True, table_format='row_models'):
    """
    Transforms the json from the automatic simulator into tables of results
    """
    if from_file is True:
        with open(results) as json_file:
            results = json.load(json_file)
    if table_format == 'row_metrics_one':
        for j in range(len(results['results'])):
            header_name = 'Model: ' + results['results'][j]['model_solver']['model'] + ', Penalization: ' + results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                header_name += ' ' + results['results'][j]['weight_calculator']['weight_technique']
            headers = [header_name, 'Value: [media, sd]']
            sorted_summary = sorted([(k, v) for k, v in results['results'][j]['summary'].items()])
            print(tabulate(sorted_summary, headers=headers))
    elif table_format == 'row_metrics_all':
        tmp_results = []
        headers = ['Metrics']
        for j in range(len(results['results'])):
            tmp_results.append(results['results'][j]['summary'])
            header_name = 'Penalization: ' + results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                header_name += ' ' + results['results'][j]['weight_calculator']['weight_technique']
            headers.append(header_name)
        one_table_result = {}
        # noinspection PyCompatibility
        for k in tmp_results[0].iterkeys():
            one_table_result[k] = tuple(d[k] for d in tmp_results)
        print(tabulate([(k,) + v for k, v in one_table_result.items()], headers=headers))
    elif table_format == 'row_models':
        table = []
        header = None
        for j in range(len(results['results'])):
            s = results['results'][j]['model_solver']['penalization']
            if 'asgl' in results['results'][j]['model_solver']['penalization']:
                s = s + '-' + results['results'][j]['weight_calculator']['weight_technique']
                if 'spca' in results['results'][j]['weight_calculator']['weight_technique']:
                    s = s + '-a' + str(results['results'][j]['weight_calculator']['spca_param']['alpha']) + '-ra' + str(results['results'][j]['weight_calculator']['spca_param']['ridge_alpha'])
            header = ['Model']
            model_table = [s]
            for key in results['results'][j]['metrics'].keys():
                model_table.append(results['results'][j]['summary'][key])
                header.append(key)
            table.append(model_table)
        print(tabulate(table, headers=header))
    return results


def combine_results(file_folder):
    """
    Take several "tmp" json files from the automatic simulator and combines the results into one final file
    """
    # Get a list of the type of models solved stored in the json files in the folder
    file_list = os.listdir(file_folder)
    identifier_list = []
    for i in range(len(file_list)):
        # Load the json file
        with open(file_folder + file_list[i]) as json_file:
            tmp_file = json.load(json_file)
        for j in range(len(tmp_file['results'])):
            # Obtain the model penalization and weight technique used
            model = tmp_file['results'][j]['model_solver']['model']
            penalization = tmp_file['results'][j]['model_solver']['penalization']
            weight_technique = ''
            if 'asgl' in penalization:
                weight_technique = tmp_file['results'][j]['weight_calculator']['weight_technique']
            identifier_list.append(str(model + penalization + weight_technique))
    unique_identifiers = np.unique(identifier_list)
    # Generate as much elements in results as types of models
    results = [None] * len(unique_identifiers)
    for i in range(len(file_list)):
        with open(file_folder + file_list[i]) as json_file:
            tmp_file = json.load(json_file)
        for j in range(len(tmp_file['results'])):
            model = tmp_file['results'][j]['model_solver']['model']
            penalization = tmp_file['results'][j]['model_solver']['penalization']
            weight_technique = ''
            if 'asgl' in penalization:
                weight_technique = tmp_file['results'][j]['weight_calculator']['weight_technique']
            identifier = model + penalization + weight_technique
            # Store the results from the loaded model into the results list
            storage_position = np.where(unique_identifiers == identifier)[0][0]
            if results[storage_position] is None:
                results[storage_position] = tmp_file['results'][j]
            else:
                # noinspection PyTypeChecker
                for key in results[storage_position]["metrics"].keys():
                    # noinspection PyTypeChecker
                    results[storage_position]['metrics'][key] = results[storage_position]['metrics'][key] + tmp_file['results'][j]['metrics'][key]
                # noinspection PyTypeChecker
                for key in results[storage_position]["extra_metrics"].keys():
                    # noinspection PyTypeChecker
                    results[storage_position]['extra_metrics'][key] = results[storage_position]['extra_metrics'][key] + tmp_file['results'][j]['extra_metrics'][key]
    # Load the parameters dictionary info
    with open(file_folder + file_list[0]) as json_file:
        tmp_file = json.load(json_file)
    parameters = tmp_file['parameters']
    final_file = dict(results=results, parameters=parameters)
    # Fill the summary dictionary
    for j in range(len(final_file['results'])):
        for key in final_file['results'][j]['metrics'].keys():
            if final_file['results'][j]['metrics'][key][0] is not None and  isinstance(final_file['results'][j]['metrics'][key][0], (int, float)):
                # noinspection PyTypeChecker
                final_file['results'][j]['summary'][key] = [round(np.mean(final_file['results'][j]['metrics'][key]), 3), round(np.std(final_file['results'][j]['metrics'][key]), 3)]
            else:
                final_file['results'][j]['summary'][key] = [None, None]
    # Check the number of models of each type solved
    number_models_solved = dict()
    n_models_list = []
    for j in range(len(final_file['results'])):
        s = final_file['results'][j]['model_solver']['penalization']
        if 'asgl' in results[j]['model_solver']['penalization']:
            s = s + '-' + final_file['results'][j]['weight_calculator']['weight_technique']
        key = final_file['results'][j]['metrics'].keys()[0]
        n_models = len(final_file['results'][j]['metrics'][key])
        number_models_solved[s] = n_models
        n_models_list.append(n_models)
    if n_models_list.count(n_models_list[0]) == len(n_models_list):
        # noinspection PyTypeChecker
        final_file['parameters']['n_repetitions'] = n_models_list[0]
    file_name = file_folder + file_list[0][:-9] + '.json'
    try:
        with open(file_name, 'w') as fp:
            json.dump(final_file, fp)
    except (TypeError or IOError):
        logger.error('Error dumping results to JSON file')
    return number_models_solved


def combine_results_var_pct(file_folder):
    """
    Take several "tmp" json files from the automatic simulator and combines the results into one final file
    """
    # Get a list of the type of models solved stored in the json files in the folder
    file_list = os.listdir(file_folder)
    identifier_list = []
    for i in range(len(file_list)):
        # Load the json file
        with open(file_folder + file_list[i]) as json_file:
            tmp_file = json.load(json_file)
        for j in range(len(tmp_file['results'])):
            # Obtain the model penalization and weight technique used
            model = tmp_file['results'][j]['model_solver']['model']
            penalization = tmp_file['results'][j]['model_solver']['penalization']
            weight_technique = ''
            var_pct = ''
            if 'asgl' in penalization:
                weight_technique = tmp_file['results'][j]['weight_calculator']['weight_technique']
                var_pct = str(tmp_file['results'][j]['weight_calculator']['var_pct'])
            identifier_list.append(str(model + penalization + weight_technique + var_pct))
    unique_identifiers = np.unique(identifier_list)
    # Generate as much elements in results as types of models
    results = [None] * len(unique_identifiers)
    for i in range(len(file_list)):
        with open(file_folder + file_list[i]) as json_file:
            tmp_file = json.load(json_file)
        for j in range(len(tmp_file['results'])):
            model = tmp_file['results'][j]['model_solver']['model']
            penalization = tmp_file['results'][j]['model_solver']['penalization']
            weight_technique = ''
            var_pct = ''
            if 'asgl' in penalization:
                weight_technique = tmp_file['results'][j]['weight_calculator']['weight_technique']
                var_pct = str(tmp_file['results'][j]['weight_calculator']['var_pct'])
            identifier = model + penalization + weight_technique + var_pct
            # Store the results from the loaded model into the results list
            storage_position = np.where(unique_identifiers == identifier)[0][0]
            if results[storage_position] is None:
                results[storage_position] = tmp_file['results'][j]
            else:
                # noinspection PyTypeChecker
                for key in results[storage_position]["metrics"].keys():
                    # noinspection PyTypeChecker
                    results[storage_position]['metrics'][key] = results[storage_position]['metrics'][key] + tmp_file['results'][j]['metrics'][key]
                # noinspection PyTypeChecker
                for key in results[storage_position]["extra_metrics"].keys():
                    # noinspection PyTypeChecker
                    results[storage_position]['extra_metrics'][key] = results[storage_position]['extra_metrics'][key] + tmp_file['results'][j]['extra_metrics'][key]
    # Load the parameters dictionary info
    with open(file_folder + file_list[0]) as json_file:
        tmp_file = json.load(json_file)
    parameters = tmp_file['parameters']
    final_file = dict(results=results, parameters=parameters)
    # Fill the summary dictionary
    for j in range(len(final_file['results'])):
        for key in final_file['results'][j]['metrics'].keys():
            if final_file['results'][j]['metrics'][key][0] is not None and  isinstance(final_file['results'][j]['metrics'][key][0], (int, float)):
                # noinspection PyTypeChecker
                final_file['results'][j]['summary'][key] = [round(np.mean(final_file['results'][j]['metrics'][key]), 3), round(np.std(final_file['results'][j]['metrics'][key]), 3)]
            else:
                final_file['results'][j]['summary'][key] = [None, None]
    # Check the number of models of each type solved
    number_models_solved = dict()
    n_models_list = []
    for j in range(len(final_file['results'])):
        s = final_file['results'][j]['model_solver']['penalization']
        if 'asgl' in results[j]['model_solver']['penalization']:
            s = s + '-' + final_file['results'][j]['weight_calculator']['weight_technique'] + '-' + str(final_file['results'][j]['weight_calculator']['var_pct'])
        key = final_file['results'][j]['metrics'].keys()[0]
        n_models = len(final_file['results'][j]['metrics'][key])
        number_models_solved[s] = n_models
        n_models_list.append(n_models)
    if n_models_list.count(n_models_list[0]) == len(n_models_list):
        # noinspection PyTypeChecker
        final_file['parameters']['n_repetitions'] = n_models_list[0]
    file_name = file_folder + file_list[0][:-9] + '.json'
    try:
        with open(file_name, 'w') as fp:
            json.dump(final_file, fp)
    except (TypeError or IOError):
        logger.error('Error dumping results to JSON file')
    return number_models_solved


def combine_results_spca(file_folder):
    """
    Take several "tmp" json files from the automatic simulator and combines the results into one final file
    """
    # Get a list of the type of models solved stored in the json files in the folder
    file_list = os.listdir(file_folder)
    identifier_list = []
    for i in range(len(file_list)):
        # Load the json file
        with open(file_folder + file_list[i]) as json_file:
            tmp_file = json.load(json_file)
        for j in range(len(tmp_file['results'])):
            # Obtain the model penalization and weight technique used
            model = tmp_file['results'][j]['model_solver']['model']
            penalization = tmp_file['results'][j]['model_solver']['penalization']
            weight_technique = ''
            alpha_spca = ''
            ridge_alpha_spca = ''
            if 'asgl' in penalization:
                weight_technique = tmp_file['results'][j]['weight_calculator']['weight_technique']
                if 'spca' in weight_technique:
                    alpha_spca = str(tmp_file['results'][j]['weight_calculator']['spca_param']['alpha']) + '-'
                    ridge_alpha_spca = str(tmp_file['results'][j]['weight_calculator']['spca_param']['ridge_alpha'])
            identifier_list.append(str(model + penalization + weight_technique + alpha_spca + ridge_alpha_spca))
    unique_identifiers = np.unique(identifier_list)
    # Generate as much elements in results as types of models
    results = [None] * len(unique_identifiers)
    for i in range(len(file_list)):
        with open(file_folder + file_list[i]) as json_file:
            tmp_file = json.load(json_file)
        for j in range(len(tmp_file['results'])):
            model = tmp_file['results'][j]['model_solver']['model']
            penalization = tmp_file['results'][j]['model_solver']['penalization']
            weight_technique = ''
            alpha_spca = ''
            ridge_alpha_spca = ''
            if 'asgl' in penalization:
                weight_technique = tmp_file['results'][j]['weight_calculator']['weight_technique']
                if 'spca' in weight_technique:
                    alpha_spca = str(tmp_file['results'][j]['weight_calculator']['spca_param']['alpha']) + '-'
                    ridge_alpha_spca = str(tmp_file['results'][j]['weight_calculator']['spca_param']['ridge_alpha'])
            identifier = model + penalization + weight_technique + alpha_spca + ridge_alpha_spca
            # Store the results from the loaded model into the results list
            storage_position = np.where(unique_identifiers == identifier)[0][0]
            if results[storage_position] is None:
                results[storage_position] = tmp_file['results'][j]
            else:
                # noinspection PyTypeChecker
                for key in results[storage_position]["metrics"].keys():
                    # noinspection PyTypeChecker
                    results[storage_position]['metrics'][key] = results[storage_position]['metrics'][key] + tmp_file['results'][j]['metrics'][key]
                # noinspection PyTypeChecker
                for key in results[storage_position]["extra_metrics"].keys():
                    # noinspection PyTypeChecker
                    results[storage_position]['extra_metrics'][key] = results[storage_position]['extra_metrics'][key] + tmp_file['results'][j]['extra_metrics'][key]
    # Load the parameters dictionary info
    with open(file_folder + file_list[0]) as json_file:
        tmp_file = json.load(json_file)
    parameters = tmp_file['parameters']
    final_file = dict(results=results, parameters=parameters)
    # Fill the summary dictionary
    for j in range(len(final_file['results'])):
        for key in final_file['results'][j]['metrics'].keys():
            if final_file['results'][j]['metrics'][key][0] is not None and  isinstance(final_file['results'][j]['metrics'][key][0], (int, float)):
                # noinspection PyTypeChecker
                final_file['results'][j]['summary'][key] = [round(np.mean(final_file['results'][j]['metrics'][key]), 3), round(np.std(final_file['results'][j]['metrics'][key]), 3)]
            else:
                final_file['results'][j]['summary'][key] = [None, None]
    # Check the number of models of each type solved
    number_models_solved = dict()
    n_models_list = []
    for j in range(len(final_file['results'])):
        s = final_file['results'][j]['model_solver']['penalization']
        if 'asgl' in results[j]['model_solver']['penalization']:
            s = s + '-' + final_file['results'][j]['weight_calculator']['weight_technique']
            if 'spca' in final_file['results'][j]['weight_calculator']['weight_technique']:
                s = s + '-' + str(final_file['results'][j]['weight_calculator']['spca_param']['alpha']) + '-' + str(final_file['results'][j]['weight_calculator']['spca_param']['ridge_alpha'])
        key = final_file['results'][j]['metrics'].keys()[0]
        n_models = len(final_file['results'][j]['metrics'][key])
        number_models_solved[s] = n_models
        n_models_list.append(n_models)
    if n_models_list.count(n_models_list[0]) == len(n_models_list):
        # noinspection PyTypeChecker
        final_file['parameters']['n_repetitions'] = n_models_list[0]
    file_name = file_folder + file_list[0][:-9] + '.json'
    try:
        with open(file_name, 'w') as fp:
            json.dump(final_file, fp)
    except (TypeError or IOError):
        logger.error('Error dumping results to JSON file')
    return number_models_solved


def extract_results(results, index, file_name=None, file_folder=None):
    num_results = len(index)
    results['parameters']['num_repetitions'] = num_results
    for i in range(len(results['results'])):
        metrics = results['results'][i]['metrics']
        extra_metrics = results['results'][i]['extra_metrics']
        for key in metrics.keys():
            metrics[key] = list(np.asarray(metrics[key])[index])
        for key in extra_metrics.keys():
            extra_metrics[key] = list(np.asarray(extra_metrics[key])[index])
        results['results'][i]['metrics'] = metrics
        results['results'][i]['extra_metrics'] = extra_metrics
    # Fill the summary
    for j in range(len(results['results'])):
        for key in results['results'][j]['metrics'].keys():
            if results['results'][j]['metrics'][key][0] is not None and  isinstance(results['results'][j]['metrics'][key][0], (int, float)):
                # noinspection PyTypeChecker
                results['results'][j]['summary'][key] = [round(np.mean(results['results'][j]['metrics'][key]), 3), round(np.std(results['results'][j]['metrics'][key]), 3)]
            else:
                results['results'][j]['summary'][key] = [None, None]
    if file_name is not None and file_folder is not None:
        file_name = file_folder + file_name + '.json'
    try:
        with open(file_name, 'w') as fp:
            json.dump(results, fp)
    except (TypeError or IOError):
        logger.error('Error dumping results to JSON file')
    return results