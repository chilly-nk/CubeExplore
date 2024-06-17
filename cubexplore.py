# import imagej

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Specify Data for Testing

import inspect

# project = none
# experiment = None
# sample = None
# data_path = None
# metadata_path = None
# correction_data_path = None

def specify_data(which = str):
  caller_globals = inspect.stack()[1][0].f_globals

  if which == 'real':
    caller_globals['project'] = '/content/drive/My Drive/DATA'
    caller_globals['experiment'] = 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve'
    caller_globals['sample'] = 'right_leg_tendon_nerve'
    caller_globals['dataset'] = 'cubes'
    
    caller_globals['data_path'] = os.path.join(project, experiment, sample, dataset)
    caller_globals['metadata_path'] = os.path.join(project, experiment, sample, 'metadata.csv')
    caller_globals['correction_data_path'] = os.path.join(project, experiment, sample, 'correction_data.lnk')

  elif which == 'test':
    caller_globals['project'] = '/content/drive/My Drive/DATA'
    caller_globals['experiment'] = 'Test Data' # the same with less cubes
    caller_globals['sample'] = 'right_leg_tendon_nerve'
    caller_globals['dataset'] = 'cubes'
    
    caller_globals['data_path'] = os.path.join(project, experiment, sample, dataset)
    caller_globals['metadata_path'] = os.path.join(project, experiment, sample, 'metadata.csv')
    caller_globals['correction_data_path'] = os.path.join(project, experiment, sample, 'correction_data')

  print('Data:', data_path)
  print('Metadata:', metadata_path)
  print('Correction Data:', correction_data_path)


# def test():
#   globals()['testvariable'] = 'nothing'
#   print(testvariable)

# mymodule.py

# def define_globals():
#     global var1, var2
#     var1 = "Hello"
#     var2 = "World"


# mymodule.py

def define_globals():
    caller_globals = inspect.stack()[1][0].f_globals
    caller_globals['var1'] = "Hello"
    caller_globals['var2'] = "World"
