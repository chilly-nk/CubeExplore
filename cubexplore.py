import os
import numpy as np
import pandas as pd

# Initiate pyimagej (at fiji mode)
import imagej
ij = imagej.init('sc.fiji:fiji')

"""LOAD CUBES"""
# Version: 2024-06-25

"""
1. Just changed the output not to be a global variable, but ruther return a dictionary with cubes data
2. Also to print the cube names while loading, to see progress
3. Changed the data to be loaded as float, not int
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

  return cubes

"""PROCESS CUBES"""
# Version 2024-06-25

"""
Introducing the alter-cube processing step
"""

def process_cubes(cubes, cubes_to_analyse, correction_data_path = None, alter_base = None, background_cube = None):

  global correction_data # might be useful to see the blip of that day
  global wavelengths_needed # just for viewing in case needed

  if background_cube:
    for cube in cubes.keys():
      cubes[cube]['data'] = cubes[cube]['data'] - cubes[background_cube]['data']

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

    for cube in cubes.keys():
      ex = cubes[cube]['ex']
      if ex in list(wavelengths_needed): #maybe this is odd, i don't remember, will check later

        correction_factor = correction_data['average'][ex]/correction_data['average'].mean()
        cubes[cube]['correction_factor'] = round(correction_factor, 2)

        data_corrected = cubes[cube]['data'] / correction_factor
        cubes[cube]['data_corrected'] = np.around(data_corrected, decimals = 2)

      else: print(f"Cube [{cube}] does not have a correction factor, but sometimes that's ok! ;)")

  if alter_base:
    for cube in cubes_to_analyse:
      altercube = cubes[cube]['data_corrected'] / (cubes[alter_base]['data_corrected'] + 1)
      cubes[cube]['alter_base'] = alter_base
      cubes[cube]['altercube'] = np.around(altercube, decimals = 4)


"""GET ROIs"""
# Version 2024-06-26 (2)

"""
Turned the function into a class, for importing and reusing convenience
Some uppercases to lowercases
"""

class Regions():

  def __init__(self):
    self.rois = {}

  def add_roi(self, region, cubes, cubes_to_analyse, y1, y2, x1, x2):
    
    self.regions[region] = {}
    
    rows = slice(y1, y2+1)
    cols = slice(x1, x2+1)

    self.regions[region]['rows'] = rows
    self.regions[region]['cols'] = cols
    self.regions[region]['rows_str'] = f'{rows.start}:{rows.stop}'
    self.regions[region]['cols_str'] = f'{cols.start}:{cols.stop}'
    self.regions[region]['num_pixels'] = (rows.stop - rows.start) * (cols.stop - cols.start)

    # Delete
    # rows = self.regions[region]['rows']
    # cols = self.regions[region]['cols']

    spectra_corrected = {}
    spectra_corrected['bycube'] = {}
    spectra_corrected['combined'] = {}

    eems_corrected = {}

    ###
    spectra = {}
    spectrum = {}

    for cube in cubes_to_analyse:

      # Get wavelengths from cubes metadata
      # This can be transferred to the cubes metadata, and later just accessed like cubes[cube]['wavelengths']
      start = cubes[cube]['em_start']
      end = cubes[cube]['em_end']
      step = cubes[cube]['step']
      wavelengths = range(start, end+1, step)

      # Slice each cube to the region's coordinates
      cube_segment = cubes[cube]['data_corrected'][rows, cols, :]

      # Reshape each cube and get 2D data for spectra for each pixel
      cube_reshaped = cube_segment.reshape(cube_segment.shape[0] * cube_segment.shape[1], cube_segment.shape[2])
      spectra[cube] = pd.DataFrame(cube_reshaped, columns = wavelengths)

      # Get the average spectra by pixels for each cube
      cube_averaged_byPixel = np.mean(cube_segment, axis = (0, 1), keepdims = True).reshape(1, cube_segment.shape[2])
      spectrum[cube] = pd.DataFrame(cube_averaged_byPixel, columns = wavelengths, index = [int(cubes[cube]['ex'])])

      # Get EEM out of averaged (over pixels) spectra, by stacking them together
      try:
        eem_averaged_corrected
      except NameError:
        eem_averaged_corrected = pd.DataFrame(columns = wavelengths)

      eem_averaged_corrected = pd.concat([eem_averaged_corrected, spectrum[cube]], axis = 0).sort_index(ascending = False)

    spectra_corrected['bycube']['all'] = spectra
    spectra_corrected['bycube']['average'] = spectrum

    self.regions[region]['eem_averaged_corrected'] = eem_averaged_corrected

    spectra_corrected['combined']['all'] = None
    spectra_corrected['combined']['average'] = None

    self.regions[region]['spectra_corrected'] = spectra_corrected 


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

