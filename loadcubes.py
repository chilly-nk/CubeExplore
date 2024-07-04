import os
import numpy as np
import pandas as pd

# Initiate pyimagej (at fiji mode)
import imagej
ij = imagej.init('sc.fiji:fiji')

"""LOAD CUBES"""
# Version: 2024-07-04

"""
Fixed metadata issue. When there was a cube, and no metadata for that (e.g. user has forgot), 
the function was not reading the metadata at all. Now it does, omitting that specific cube,
for which metadata is not specified.
"""

def load_cubes_im3(data_path, metadata_path = None):

  cube_names = sorted(os.listdir(data_path))
  cubes = {}
  for i in cube_names:
    # cubes[i] = tiff.imread(os.path.join(data_path, i)).transpose(1, 2, 0) # for loading cubes from 3D tiffs
    print(f'Loading {i}...')
    img = ij.io().open(os.path.join(data_path, i))
    cube = np.array(ij.py.from_java(img), dtype = float)
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
    metadata = pd.read_csv(metadata_path)
    metadata['excitation'] = metadata.excitation.astype(str)
    metadata.set_index('excitation', inplace = True)
    for cube in cubes.keys():
      ex = cube[:-4]
      cubes[cube]['ex'] = round(float(ex), 1) if ex.isdigit() else ex
      if ex not in metadata.index:
        print(f'Condition "{ex}" not in metadata.')
        continue
      cubes[cube]['em_start'] = int(metadata.loc[ex, 'emission_start'])
      cubes[cube]['em_end'] = int(metadata.loc[ex, 'emission_end'])
      cubes[cube]['step'] = int(metadata.loc[ex, 'step'])
      exp = metadata.loc[ex, 'exp']
      cubes[cube]['expos_val'] = float(exp) if str(exp).isdigit() else exp
      cubes[cube]['notes'] = metadata.loc[ex, 'notes']

  return cubes


# Version: 2024-06-25

"""
1. Just changed the output not to be a global variable, but ruther return a dictionary with cubes data
2. Also to print the cube names while loading, to see progress
3. Changed the data to be loaded as float, not int
"""

# def load_cubes_im3(data_path, metadata_path = None):

#   cube_names = sorted(os.listdir(data_path))
#   cubes = {}
#   for i in cube_names:
#     # cubes[i] = tiff.imread(os.path.join(data_path, i)).transpose(1, 2, 0) # for loading cubes from 3D tiffs
#     print(f'Loading {i}...')
#     img = ij.io().open(os.path.join(data_path, i))
#     cube = np.array(ij.py.from_java(img), dtype = float)
#     ex = i[:-4]
#     cube_data = {
#                  'ex': ex,
#                  'em_start': None,
#                  'em_end': None,
#                  'step': None,
#                  'num_rows': cube.shape[0],
#                  'num_cols': cube.shape[1],
#                  'num_bands': cube.shape[2],
#                  'expos_val': None,
#                  'notes': None,
#                  'data': cube,
#                 }

#     cubes[i] = cube_data

#   if metadata_path:
#     with open(metadata_path, 'r') as file:
#       metadata = file.readlines()[1:]
#       if len(metadata) == len(cubes.keys()):
#         for cube, line in zip(cubes.keys(), metadata):
#           ex, em_start, em_end, step, exp, note = line.strip().split(',')
#           cubes[cube]['ex'] = round(float(ex), 1) if ex.isdigit() else ex
#           cubes[cube]['em_start'] = int(em_start)
#           cubes[cube]['em_end'] = int(em_end)
#           cubes[cube]['step'] = int(step)
#           cubes[cube]['expos_val'] = float(exp) if exp.replace('.', '').replace('-','').isdigit() else exp
#           cubes[cube]['notes'] = note
#       else:
#         print('Warning! Metadata and cubes do not match')

#   return cubes