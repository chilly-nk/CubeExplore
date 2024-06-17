import imagej

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# Specify Data for Testing
def specify_data(which = str):

  global project
  global experiment
  global sample
  global data_path
  global metadata_path
  global correction_data_path

  if which == 'real':
    project = '/content/drive/My Drive/DATA'
    experiment = 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve'
    # experiment = 'Test Data' # the same with less cubes
    sample = 'right_leg_tendon_nerve'
    dataset = 'cubes'

    data_path = os.path.join(project, experiment, sample, dataset)
    metadata_path = os.path.join(project, experiment, sample, 'metadata.csv')
    correction_data_path = os.path.join(project, experiment, sample, 'correction_data.lnk')

  elif which == 'test':
    project = '/content/drive/My Drive/DATA'
    # experiment = 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve'
    experiment = 'Test Data' # the same with less cubes
    sample = 'right_leg_tendon_nerve'
    dataset = 'cubes'

    data_path = os.path.join(project, experiment, sample, dataset)
    metadata_path = os.path.join(project, experiment, sample, 'metadata.csv')
    correction_data_path = os.path.join(project, experiment, sample, 'correction_data')

  print('Data:', data_path)
  print('Metadata:', metadata_path)
  print('Correction Data:', correction_data_path)