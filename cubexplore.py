# import imagej

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""TEST Data"""
# Version 2024-06-24

project = '/content/drive/My Drive/DATA'
experiment = 'Test Data' # the same with less cubes
sample = 'right_leg_tendon_nerve'
dataset = 'cubes'

test_data = {
    'project' : '/content/drive/My Drive/DATA',
    'experiment' : 'Test Data',
    'sample' : 'right_leg_tendon_nerve',
    'dataset' : 'cubes',
    'data_path' : os.path.join(project, experiment, sample, dataset),
    'metadata_path' : os.path.join(project, experiment, sample, 'metadata.csv'),
    'correction_data_path' : os.path.join(project, experiment, sample, 'correction_data'),
}

project = '/content/drive/My Drive/DATA'
experiment = 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve'
sample = 'right_leg_tendon_nerve'
dataset = 'cubes'

real_data = {
    'project' : '/content/drive/My Drive/DATA',
    'experiment' : 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve',
    'sample' : 'right_leg_tendon_nerve',
    'dataset' : 'cubes',
    'data_path' : os.path.join(project, experiment, sample, dataset),
    'metadata_path' : os.path.join(project, experiment, sample, 'metadata.csv'),
    'correction_data_path' : os.path.join(project, experiment, sample, 'correction_data.lnk'),
}

some_data = {'test': test_data, 'real': real_data}

# Version 2024-06-17

# import inspect

# # project = none
# # experiment = None
# # sample = None
# # data_path = None
# # metadata_path = None
# # correction_data_path = None

# def some_data(which = str):
#   caller_globals = inspect.stack()[1][0].f_globals

#   if which == 'real':
    
#     project = '/content/drive/My Drive/DATA'
#     experiment = 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve'
#     sample = 'right_leg_tendon_nerve'
#     dataset = 'cubes'
#     data_path = os.path.join(project, experiment, sample, dataset)
#     metadata_path = os.path.join(project, experiment, sample, 'metadata.csv')
#     correction_data_path = os.path.join(project, experiment, sample, 'correction_data.lnk')
    
#     caller_globals['project'] = project
#     caller_globals['experiment'] = experiment
#     caller_globals['sample'] = sample
#     caller_globals['dataset'] = dataset
#     caller_globals['data_path'] = data_path
#     caller_globals['metadata_path'] = metadata_path
#     caller_globals['correction_data_path'] = correction_data_path

#   elif which == 'test':
    
#     project = '/content/drive/My Drive/DATA'
#     experiment = 'Test Data' # the same with less cubes
#     sample = 'right_leg_tendon_nerve'
#     dataset = 'cubes'
#     data_path = os.path.join(project, experiment, sample, dataset)
#     metadata_path = os.path.join(project, experiment, sample, 'metadata.csv')
#     correction_data_path = os.path.join(project, experiment, sample, 'correction_data')
    
#     caller_globals['project'] = project
#     caller_globals['experiment'] = experiment
#     caller_globals['sample'] = sample
#     caller_globals['dataset'] = dataset
    
#     caller_globals['data_path'] = data_path
#     caller_globals['metadata_path'] = metadata_path
#     caller_globals['correction_data_path'] = correction_data_path

#   print('Data:', data_path)
#   print('Metadata:', metadata_path)
#   print('Correction Data:', correction_data_path)


# # def test():
# #   globals()['testvariable'] = 'nothing'
# #   print(testvariable)

# # mymodule.py

# # def define_globals():
# #     global var1, var2
# #     var1 = "Hello"
# #     var2 = "World"


# # mymodule.py

# def define_globals():
#     caller_globals = inspect.stack()[1][0].f_globals
#     caller_globals['var1'] = "Hello"
#     caller_globals['var2'] = "World"

""""""
