import os
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pytz
import spectral as spy
import spectral.io.envi as envi
from datetime import datetime
from sklearn.decomposition import PCA

# Initiate pyimagej (at fiji mode)
import imagej
ij = imagej.init('sc.fiji:fiji')

class Cubes:
  def __init__(self, data_path, metadata_path = None, cubes_to_load = None, data_source = 'nuance'):
    
    yerevantime = pytz.timezone('Asia/Yerevan')
    time = datetime.now().astimezone(yerevantime).strftime('%Y-%m-%d %H:%M:%S')
    self.last_loaded = time
    self.data_source = data_source
    
    self.raw = {}
    self.metadata = {}
    self.processed = {}
    self.normalized = {} # Not ready yet
    self.combined = None # Not ready yet

    self.selected_rows = None
    self.selected_cols = None
    
    self.pcs_bycube = {}
    self.pcs_bycube_transformed = {}
    
    if cubes_to_load:
      cube_names = sorted(cubes_to_load)
    else:
      cube_names = sorted(os.listdir(data_path))
    self.names = cube_names
    
    for cubename in cube_names:
      # cubes[i] = tiff.imread(os.path.join(data_path, i)).transpose(1, 2, 0) # for loading cubes from 3D tiffs
      print(f'Loading {cubename}...')
      if data_source == 'nuance':
        img = ij.io().open(os.path.join(data_path, cubename))
        img_loaded = ij.py.from_java(img)
      elif data_source == 'goldeneye' or data_source == 'snapshot':
        basename = cubename.split("_")[0].split(".")[0]
        header_file = os.path.join(data_path, cubename,  f'{basename}_processed_image.hdr')
        data_file = os.path.join(data_path, cubename, f'{basename}_processed_image.bin')
        wavelengths_file = os.path.join(data_path, cubename, f'{basename}_wavelengths.csv')
        wavelengths = round(pd.read_csv(wavelengths_file).T.reset_index().T.astype(float).reset_index(drop = True)).astype(int)
        wavelengths = np.array(wavelengths[0])
        img = envi.open(header_file, data_file)
        img_loaded = img.load()
      cube = np.array(img_loaded, dtype = np.float32)
      
      self.raw[cubename] = cube
      
      ex = cubename.split("_")[0].split(".")[0]
      md = {'ex': ex,
            'em_start': None,
            'em_end': None,
            'step': None,
            'num_rows': cube.shape[0],
            'num_cols': cube.shape[1],
            'num_bands': cube.shape[2],
            'expos_val': None,
            'notes': None,
            'wavelengths': wavelengths if 'wavelengths' in locals() else None,
            }

      self.metadata[cubename] = md

    if metadata_path:
      metadata = pd.read_csv(metadata_path)
      metadata['excitation'] = metadata.excitation.astype(str)
      metadata.set_index('excitation', inplace = True)
      self.metadata_df = metadata
      
      for cubename in self.metadata.keys():
        ex = self.metadata[cubename]['ex']
        self.metadata[cubename]['ex'] = round(float(ex), 1) if ex.isdigit() else ex
        if ex not in metadata.index:
          print(f"Attention! User has not provided metadata for cube '{ex}'.")
          continue
        em_start = int(metadata.loc[ex, 'emission_start'])
        em_end = int(metadata.loc[ex, 'emission_end'])
        step = int(metadata.loc[ex, 'step'])
        self.metadata[cubename]['em_start'] = em_start
        self.metadata[cubename]['em_end'] = em_end
        self.metadata[cubename]['step'] = step
        exp = metadata.loc[ex, 'exp']
        self.metadata[cubename]['expos_val'] = float(exp) if str(exp).isdigit() else exp
        self.metadata[cubename]['notes'] = metadata.loc[ex, 'notes']
        self.metadata[cubename]['wavelengths'] = np.array(range(em_start, em_end+1, step))

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
        print(f"Correcting cube '{cube}'...")
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
    
    if self.data_source == 'goldeneye' or self.data_source == 'snapshot':
      blue_bands = range(11, 21)
      green_bands = range(32, 42)
      red_bands = range(52, 62)
    # But what if you want to provide bands even when its 'snapshot'. They will be overwritten here. Need to fix this.

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
    coords = pd.Series([y1, y2, x1, x2])
    if coords.notna().all():
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

  def reshape_bycube(self, which_data = 'processed'):
    data_to_process = getattr(self, which_data)
    self.reshaped_bycube_input = which_data
    for cubename in data_to_process.keys():
      cube = data_to_process[cubename]
      cube_reshaped = np.reshape(cube, (cube.shape[0]*cube.shape[1], cube.shape[2]))
      self.reshaped[cubename] = cube_reshaped

  def get_pcs_bycube(self, components = 3, which_data = 'processed', extra_transform = True, trans_factor = 0.5, trans_inplace = False):
    data_to_process = getattr(self, which_data)
    for cubename in data_to_process:
      cube = data_to_process[cubename]
      cube_reshaped = cube.reshape(cube.shape[0]*cube.shape[1], cube.shape[2]).astype(np.float64)
      pca = PCA(n_components = components)
      PCs = pca.fit_transform(cube_reshaped)
      
      if extra_transform == True:
        PCs_transformed = np.sign(PCs) * np.abs(PCs) ** trans_factor
        self.pcs_bycube_transformed[cubename] = PCs_transformed
      
      if trans_inplace == True:
        continue
      else:
        self.pcs_bycube[cubename] = PCs

  def normalize(self, which = 'processed'):
    data_to_process = getattr(self, which)
    for cubename in data_to_process.keys():
      cube = data_to_process[cubename]
      cube_max = np.max(cube, axis = 2, keepdims = True) + np.finfo(float).eps
      cube_normalized = cube / cube_max
      self.normalized[cubename] = cube_normalized

  def quick_eem(self, cubes_to_analyse = None, which_from = 'processed'):
    
    data_to_process = getattr(self, which_from)
    if cubes_to_analyse:
      cube_names = cubes_to_analyse
    else:
      cube_names = self.names

    rows = self.selected_rows
    cols = self.selected_cols
    eem = pd.DataFrame()
    for cubename in cube_names:
      cube_segment = data_to_process[cubename][rows, cols, :]
      cube_segment_avg = np.mean(cube_segment, axis = (0, 1), keepdims = True).reshape(1, cube_segment.shape[2])
      ex = pd.Series(str(self.metadata[cubename]['ex']))
      wavelengths = self.metadata[cubename]['wavelengths']
      spectrum_df = pd.DataFrame(cube_segment_avg, columns = wavelengths, index = ex)
      eem = pd.concat([eem, spectrum_df], axis = 0)
    eem = eem.sort_index(ascending = False)
    self.last_eem = eem
    sns.heatmap(eem, cmap = 'coolwarm')
    plt.title('Average EEM of Selected Region')
    plt.xlabel('Em.')
    plt.ylabel('Ex.')
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)


  # This doesn't work yet
  # def savefile(self, name = 'cubes', path = str):
  #   filepath = os.path.join(path, name + '.pkl')
  #   with open(filepath, 'wb') as file:
  #     pickle.dump(self, file)
