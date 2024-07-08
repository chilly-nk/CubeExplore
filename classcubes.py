import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Initiate pyimagej (at fiji mode)
import imagej
ij = imagej.init('sc.fiji:fiji')

class Cubes:
  
  def __init__(self, data_path, metadata_path = None, cubes_to_load = None):
    
    self.last_loaded = datetime.datetime.now()
    
    self.raw = {}
    self.metadata = {}
    self.processed = {}
    self.normalized = {}
    
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

  def view(self, cube_to_view: str, y1 = None, y2 = None, x1 = None, x2 = None, blue_bands = range(3, 9), green_bands = range(13, 19), red_bands = range(23, 29)):
    cube = self.raw[cube_to_view]

    # Extract data for each channel
    red_data = np.mean(cube[:, :, red_bands], axis=-1)
    green_data = np.mean(cube[:, :, green_bands], axis=-1)
    blue_data = np.mean(cube[:, :, blue_bands], axis=-1)

    # Normalize the data to [0, 1]
    normalized_red = (red_data - np.min(red_data)) / (np.max(red_data) - np.min(red_data))
    normalized_green = (green_data - np.min(green_data)) / (np.max(green_data) - np.min(green_data))
    normalized_blue = (blue_data - np.min(blue_data)) / (np.max(blue_data) - np.min(blue_data))

    # Stack the channels to create an RGB image
    rgb_image = np.stack([normalized_red, normalized_green, normalized_blue], axis=-1)  

    # Display the RGB image
    plt.figure(figsize = (10, 10))
    ax = plt.imshow(rgb_image);
    
    # Set the boundaries
    coords = [y1, y2, x1, x2]
    if all(coords):
      plt.axvline(x = x1, color = 'red', linewidth = 0.5, linestyle = '--');
      plt.axvline(x = x2, color = 'red', linewidth = 0.5, linestyle = '--');
      plt.axhline(y = y1, color = 'red', linewidth = 0.5, linestyle = '--');
      plt.axhline(y = y2, color = 'red', linewidth = 0.5, linestyle = '--');
    
      self.selected_rows = slice(min(y1, y2), max(y1, y2)+1)
      self.selected_cols = slice(min(x1, x2), max(x1, x2)+1)
    # elif any(coords):
    #   coords_dict = {
    #     'y1': y1,
    #     'y2': y2,
    #     'x1': x1,
    #     'x2': x2,
    #   }
    #   not_provided = [key for key, value in coords_dict.items() if value is None]
    #   print(f"{not_provided} not provided")
    # else: print('No coordinates provided for defining a region.\nIf you want you can provide y1, y2, x1, x2.')
    
    plt.show()

  def crop(self, y1 = None, y2 = None, x1 = None, x2 = None):
    coords = [y1, y2, x1, x2]
    if all(coord is None for coord in coords):
      try:
        rows = self.selected_rows
        cols = self.selected_cols
      except:
        print("Oops! Seems you haven't specified any coordinates.")
    elif all(coords):
      rows = slice(min(y1, y2), max(y1, y2)+1)
      cols = slice(min(x1, x2), max(x1, x2)+1)
    else:
      print('Oops! Seems you have missed some of the coordinates (y1, y2, x1, x2).')

    for cubename in self.raw.keys():
      cube = self.raw[cubename]
      cube_cropped = cube[rows, cols, :]
      self.raw[cubename] = cube_cropped
      self.metadata[cubename]['num_rows'] = cube_cropped.shape[0]
      self.metadata[cubename]['num_cols'] = cube_cropped.shape[1]

    if self.processed:
      for cubename in self.processed.keys():
        cube = self.processed[cubename]
        cube_cropped = cube[rows, cols, :]
        self.processed[cubename] = cube_cropped

    if self.normalized:
      for cubename in self.normalized.keys():
        cube = self.normalized[cubename]
        cube_cropped = cube[rows, cols, :]
        self.normalized[cubename] = cube_cropped
    