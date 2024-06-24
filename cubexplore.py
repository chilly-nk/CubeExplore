# import imagej

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""LOAD CUBES"""
# Version: 2024-06-24

def load_cubes_im3(data_path, metadata_path = None, correction_data_path = None):

  cube_names = sorted(os.listdir(data_path))
  cubes = {}
  for i in cube_names:
    # cubes[i] = tiff.imread(os.path.join(data_path, i)).transpose(1, 2, 0) # for loading cubes from 3D tiffs
    img = ij.io().open(os.path.join(data_path, i))
    cube = np.array(ij.py.from_java(img))
    ex = i[:-4]
    cube_data = {
                 'ex': ex,
                 'em_start': None,
                 'em_end': None,
                 'step': None,
                 'num_rows': cube.shape[0],
                 'num_cols': cube.shape[1],
                 'num_bands': cube.shape[2],
                 'expos_val': None,
                 'notes': None,
                 'data': cube,
                }

    cubes[i] = cube_data

  if metadata_path:
    with open(metadata_path, 'r') as file:
      metadata = file.readlines()[1:]
      if len(metadata) == len(cubes.keys()):
        for cube, line in zip(cubes.keys(), metadata):
          ex, em_start, em_end, step, exp, note = line.strip().split(',')
          cubes[cube]['ex'] = round(float(ex), 1) if ex.isdigit() else ex
          cubes[cube]['em_start'] = int(em_start)
          cubes[cube]['em_end'] = int(em_end)
          cubes[cube]['step'] = int(step)
          cubes[cube]['expos_val'] = float(exp) if exp.replace('.', '').replace('-','').isdigit() else exp
          cubes[cube]['notes'] = note
      else:
        print('Warning! Metadata and cubes do not match')

  if correction_data_path:
    globals()['correction_data_path'] = correction_data_path # Getting it here (for desktop user convenience), but will be used in process_cubes() function

  return cubes

"""PROCESS CUBES"""
# Version 2024-06-17

def process_cubes(cubes, cubes_to_analyse, correction_data_path = None):

  global correction_data # might be useful to see the blip of that day
  global wavelengths_needed # just for viewing in case needed

  if correction_data_path:

    # Load Correction Data (TLS Basic Wavelength Scan, several scans repetitions). All scans must have the same start, stop, step
    file_num = [int(filename[:-4]) for filename in os.listdir(correction_data_path)]
    anyfile_for_wavelengths = 1
    anyfile = pd.read_csv(os.path.join(correction_data_path, str(anyfile_for_wavelengths) + '.TRQ'), sep = '\t', skiprows = 9)
    wavelengths = round(anyfile.X, 1).astype(float)

    correction_data = pd.DataFrame({'wavelength': wavelengths})
    measurements = [] #seems to be not useful, maybe we can eliminate later

    for num in file_num:
      path = os.path.join(correction_data_path, str(num)+'.TRQ')
      df = pd.read_csv(path, sep = '\t', skiprows = 9)
      y = df.Y * 10 ** 6
      colname = 'm' + str(num)
      measurements += [colname]
      # correction_data = pd.concat([correction_data, y], axis = 1)
      correction_data.loc[:, colname] = y

    correction_data.set_index(keys = 'wavelength', inplace = True)

    # Select only those wavelengths which are corresponding tou our cubes of interest
    wavelengths_needed = [cubes[cube]['ex'] for cube in cubes_to_analyse if cubes[cube]['ex'] in list(wavelengths)]
    correction_data = correction_data.loc[wavelengths_needed]

    correction_data['average'] = correction_data[measurements].mean(axis = 1)
    correction_data_average = correction_data['average']

    for cube in cubes.keys():
      ex = cubes[cube]['ex']
      if ex in list(wavelengths_needed): #maybe this is odd, i don't remember, will check later

        correction_factor_byMean = correction_data_average[ex]/correction_data_average.mean()
        cubes[cube]['correction_factor_byMean'] = round(correction_factor_byMean, 2)

        data_corrected_byMean = cubes[cube]['data'] / correction_factor_byMean
        cubes[cube]['data_corrected_byMean'] = np.around(data_corrected_byMean, decimals = 2)

      else: print(f"Cube '{cube}' does not have a correction factor, but sometimes that's ok! ;)")


"""TEST DATA"""
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
