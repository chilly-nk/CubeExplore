import os
import numpy as np
import pandas as pd
import datetime

# Initiate pyimagej (at fiji mode)
import imagej
ij = imagej.init('sc.fiji:fiji')

class Cubes:
  
  def __init__(self, data_path, metadata_path = None, cubes_to_load = None):
    
    self.last_loaded = datetime.datetime.now()
    
    self.raw = {}
    self.metadata = {}
    self.processed = {}
    
    if cubes_to_load:
      cube_names = cubes_to_load
    else:
      cube_names = sorted(os.listdir(data_path))
    
    for i in cube_names:
      # cubes[i] = tiff.imread(os.path.join(data_path, i)).transpose(1, 2, 0) # for loading cubes from 3D tiffs
      print(f'Loading {i}...')
      img = ij.io().open(os.path.join(data_path, i))
      cube = np.array(ij.py.from_java(img), dtype = np.float32)
      self.raw[i] = cube
      
      ex = i[:-4]
      md = {'ex': ex,
            'em_start': None,
            'em_end': None,
            'step': None,
            'num_rows': cube.shape[0],
            'num_cols': cube.shape[1],
            'num_bands': cube.shape[2],
            'expos_val': None,
            'notes': None,
            }

      self.metadata[i] = md

    if metadata_path:
      metadata = pd.read_csv(metadata_path)
      metadata['excitation'] = metadata.excitation.astype(str)
      metadata.set_index('excitation', inplace = True)
      self.metadata_df = metadata
      
      for cube in self.metadata.keys():
        ex = cube[:-4]
        self.metadata[cube]['ex'] = round(float(ex), 1) if ex.isdigit() else ex
        if ex not in metadata.index:
          print(f"Attention! Cube '{ex}' is not provided with metadata by the user.")
          continue
        self.metadata[cube]['em_start'] = int(metadata.loc[ex, 'emission_start'])
        self.metadata[cube]['em_end'] = int(metadata.loc[ex, 'emission_end'])
        self.metadata[cube]['step'] = int(metadata.loc[ex, 'step'])
        exp = metadata.loc[ex, 'exp']
        self.metadata[cube]['expos_val'] = float(exp) if str(exp).isdigit() else exp
        self.metadata[cube]['notes'] = metadata.loc[ex, 'notes']

  def process(self, cubes_to_analyse, background_cube = None, correction_data_path = None):
    if background_cube:
      for cube in cubes_to_analyse:
        print(f"Subtracting background from '{cube}'...")
        cube_subtracted = self.raw[cube] - self.raw[background_cube]
        negatives = (cube_subtracted < 0)
        cube_subtracted[negatives] = 0
        self.processed[cube] = cube_subtracted
      print('Background subtraction done.\n See subtracted cubes in cubes.processed attribute.\n--------------------------------')
    else: print('Attention! No background subtraction took place.\n--------------------------------')

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

      # Select only those wavelengths which are corresponding to our cubes of interest
      wavelengths_needed = [self.metadata[cube]['ex'] for cube in cubes_to_analyse if self.metadata[cube]['ex'] in list(wavelengths)]
      correction_data = correction_data.loc[wavelengths_needed]

      correction_data['average'] = correction_data[measurements].mean(axis = 1)
      self.correction_data = correction_data
      self.wavelengths_needed = wavelengths_needed

      if not self.processed:
        cubes_to_correct = self.raw.keys()
      else:
        cubes_to_correct = self.processed.keys()

      self.cubes_to_correct = cubes_to_correct
      for cube in cubes_to_correct:
        print(f"Correction of cube '{cube}'...")
        ex = self.metadata[cube]['ex']
        if ex in list(wavelengths_needed): #maybe this is odd, i don't remember, will check later

          correction_factor = correction_data['average'][ex]/correction_data['average'].mean()
          self.metadata[cube]['correction_factor'] = round(correction_factor, 2)

          if background_cube:
            data_corrected = self.processed[cube] / correction_factor
          else:
            data_corrected = self.raw[cube] / correction_factor
          self.processed[cube] = np.around(data_corrected, decimals = 2)

        else: print(f"Cube '{cube}' does not have a correction factor, but sometimes that's ok! ;)") 

      print('Correction of cubes done. See corrected cubes in cubes.processed attribute.\n--------------------------------')
    else: print('Attention! No data correction took place.\n--------------------------------') 