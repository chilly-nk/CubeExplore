# Basic modules
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

# General modules
import json
import pytz
import datetime
from datetime import datetime
from typing import Optional
from collections import defaultdict

# Image analysis modules
import tifffile as tiff
import spectral.io.envi as envi
from PIL import Image

# Data processing modules
from scipy.ndimage import gaussian_filter as gf

# ML modules
from sklearn.decomposition import PCA

# CubeExplore functions
from .qc import SampleNames
from .utils import ensure_list
from .utils import bin_mask


class Cubes:
  def __init__(self, data_path, metadata_path=None, sample_id=None, cubes_to_load=None, data_source='tiff_cubes'):
    
    time = self.time()
    self.log = {}
    self.log[time] = {}
    self.log[time]['action'] = 'Dataset loaded.'
    
    self.data_source = data_source
    self.log[time]['data_source'] = data_source
    self.data_path = data_path
    self.log[time]['data_path'] = data_path
    self.metadata_path = metadata_path
    self.log[time]['metadata_path'] = metadata_path
    self.log[time]['cubes_loaded'] = cubes_to_load if cubes_to_load is not None else 'All'

    self.folder = os.path.basename(self.data_path)
    sample_names = SampleNames(os.path.dirname(self.data_path)).ref_samples
    samples = [samplename if samplename in self.folder else None for samplename in sample_names]
    self.sample = samples[0] if len(samples) > 0 else None
    
    self.cubes_to_analyse = None
    
    self.raw = {}
    self.raw_info = defaultdict(dict) # new way of storing related data
    self.size = () # to store the y*x size of the cubes, after checking that all cubes are consistent
    self.metadata = {}
    self.tls_spectrum = None
    self.spectral_sensitivity = None
    self.processed = {}
    self.processed_info = {}
    self.normalized = {}
    self.normalized_info = {}
    
    self.combined = {}
    self.combined_info = defaultdict(dict) # new 2025-05-25
    # self.combined_wvls = {} # maybe delete
    self.combined_metadata = {}
    
    self.reshaped = {}

    self.averaged = {}
    self.summed = {}

    self.color_bands = {}
    self.color_bands['nuance'] = {'red': range(23, 29), 'green': range(13, 19), 'blue': range(3, 9)}
    self.color_bands['goldeneye'] = {'red': range(52, 62), 'green': range(32, 42), 'blue': (11, 21)}
    
    self.selected_rows = None
    self.selected_cols = None
    
    self.reset_rois()
    
    self.pcs = {}
    self.pcs_transformed = {}

    self.scaled_data = {}

    self.mask = None
    self.mask_labels = {}

    self.spectra = {} # By Cube
    self.spectra_avg = {} # By Cube
    self.spectra_combined = None # In progress
    self.spectra_combined_avg = None

    if metadata_path:
      self.metadata_df = read_metadata(metadata_path, sample_id)
    
    if cubes_to_load:
      cube_names = sorted(cubes_to_load)
    else:
      cube_names = sorted(os.listdir(data_path))
    self.names = cube_names

    if data_source == 'nuance':
      from .imagej_init import load_imagej
      ij = load_imagej()
    
    for cubename in cube_names:
      print(f"Loading '{cubename}'...")
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
      elif data_source == 'tiff_cubes':
        img_loaded = tiff.imread(os.path.join(data_path, cubename)).transpose(1, 2, 0)      
      elif data_source == 'tiff_slices':
        img_loaded = read_cube_slices(data_path, cubename)
        
      cube = np.array(img_loaded, dtype = np.float32)
      
      self.raw[cubename] = cube
      self.raw_info[cubename].update({'wvls': np.arange(cube.shape[2])})
      
      ex = cubename.split("-")[0].split(".")[0]
      md = {'ex': ex,
            'emission_start': None,
            'emission_end': None,
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
      # metadata = pd.read_csv(metadata_path)
      # metadata['excitation'] = metadata.excitation.astype(str)
      # metadata.set_index('excitation', inplace = True)
      # self.metadata_df = metadata
      
      for cubename in self.metadata.keys():
        ex = self.metadata[cubename]['ex'] # Excitation only is used, because in Snapshot data cubenames are too long, so only the first component - before the underscore - is used.
        # self.metadata[cubename]['ex'] = round(float(ex), 1) if ex.isdigit() else ex
        if ex not in self.metadata_df.index:
          print(f"Attention! User has not provided metadata for cube '{ex}'")
          continue
        emission_start = int(self.metadata_df.loc[ex, 'emission_start_nm'])
        emission_end = int(self.metadata_df.loc[ex, 'emission_end_nm'])
        step = int(self.metadata_df.loc[ex, 'step_nm'])
        self.metadata[cubename]['emission_start'] = emission_start
        self.metadata[cubename]['emission_end'] = emission_end
        self.metadata[cubename]['step'] = step
        exp = self.metadata_df.loc[ex, 'exposure_time_ms']
        self.metadata[cubename]['expos_val'] = float(exp) if str(exp).isdigit() else exp
        # self.metadata[cubename]['notes'] = self.metadata_df.loc[ex, 'notes'] # Causes a bug when there is no 'notes' field in the metadata
        self.metadata[cubename]['wavelengths'] = np.array(range(emission_start, emission_end+1, step))

        # Developing better and more relevant internal metadata (except the metadata_df) for each cube
        self.raw_info[cubename].update({'wvls': self.get_wvls(cubename)})

    self.sizes = []
    for cubename in self.raw.keys():
      cube = self.raw[cubename]
      size = cube.shape[:2]
      self.sizes.append(size)
    if all([size == self.sizes[0] for size in self.sizes]):
      self.size = self.sizes[0]
    else:
      print("Attention! Not all cubes have the same size in xy axes:")
      for i, cubename in enumerate(self.raw.keys()):
        print(f"{cubename}: {self.sizes[i]}")


  def read_tls_data(self, tls_spectrum_path):
    # Load Correction Data (TLS Basic Wavelength Scan, several scans repetitions). All scans must have the same start, stop, step
    self.tls_spectrum = None
    self.tls_spectrum_path = tls_spectrum_path
    data_files = os.listdir(tls_spectrum_path)
    for filename in data_files:
      measurement = filename.split('.')[0]
      data = pd.read_csv(os.path.join(tls_spectrum_path, filename), sep = '\t', skiprows = 9)
      wavelength = round(data.X)
      optical_power = data.Y * 10 ** 6
      data_new = pd.DataFrame({'Wavelength': wavelength, f'OP_uW_m{measurement}': optical_power})
      data_new = data_new.reset_index(drop = True).set_index('Wavelength')

      if self.tls_spectrum is None:
        self.tls_spectrum = data_new
      else:
        self.tls_spectrum = pd.concat([self.tls_spectrum, data_new], axis = 1)

    self.tls_spectrum['Average'] = self.tls_spectrum.mean(axis = 1)
    self.tls_spectrum_all = self.tls_spectrum

    wavelengths_digit = [self.metadata[cubename]['ex'] for cubename in self.names if self.metadata[cubename]['ex'].isdigit()]
    wavelengths_needed = [float(wvl) for wvl in wavelengths_digit if float(wvl) in self.tls_spectrum.index]
    self.tls_spectrum = self.tls_spectrum.loc[wavelengths_needed]
    return self

  def get_spectral_sensitivity(self, spectral_sensitivity_data):
    self.spectral_sensitivity = pd.read_csv(spectral_sensitivity_data, index_col = 0)
  
  def join_tls_data(self, id_col='light_power_uW'):
    tls_data= self.tls_spectrum_all.copy()['Average']
    tls_data.index = tls_data.index.astype(int).astype(str)
    tls_data.name = id_col
    self.metadata_df.update(tls_data, join='left')
    return self

  def save_metadata(self, metadata_path=None):
    if metadata_path==None:
      metadata_path = self.metadata_path
    self.metadata_df.to_csv(metadata_path)

  def process(self, cubes_to_analyse, background_cube = None, tls_spectrum_path = None, spectral_sensitivity_data = None):
    
    if background_cube:
      for cube in cubes_to_analyse:
        print(f"Subtracting background from '{cube}'...")
        cube_subtracted = self.raw[cube] - self.raw[background_cube]
        negatives = (cube_subtracted < 0)
        cube_subtracted[negatives] = 0
        self.processed[cube] = cube_subtracted
      print('Background subtraction done. See subtracted cubes in cubes.processed attribute.\n--------------------------------')
    else: print('Attention! No background subtraction took place.\n--------------------------------')

    if tls_spectrum_path:
      self.read_tls_data(tls_spectrum_path)

      if not self.processed:
        cubes_to_correct = self.raw.keys()
      else:
        cubes_to_correct = self.processed.keys()

      self.cubes_to_correct = cubes_to_correct
      for cubename in cubes_to_correct:
        print(f"Correcting by light source cube '{cubename}'...")
        ex = self.metadata[cubename]['ex']
        ex = float(ex) if ex.isdigit() else ex
        if ex in self.tls_spectrum.index:

          correction_factor = self.tls_spectrum['Average'][ex]/self.tls_spectrum['Average'].mean()
          self.metadata[cubename]['correction_factor'] = round(correction_factor, 2)

          if background_cube:
            data_corrected = self.processed[cubename] / correction_factor
          else:
            data_corrected = self.raw[cubename] / correction_factor
          self.processed[cubename] = np.around(data_corrected, decimals = 2)

        else: print(f"Cube '{cubename}' does not have a correction factor, but sometimes that's ok! ;)") 
      print('Correction of cubes by light source done. See corrected cubes in attribute self.processed .\n--------------------------------')
    else: print('Attention! No correction by light source took place.\n--------------------------------') 
    
    if spectral_sensitivity_data:
      self.get_spectral_sensitivity(spectral_sensitivity_data)
      
      if not self.processed:
        cubes_to_correct = self.raw.keys()
      else:
        cubes_to_correct = self.processed.keys()

      self.cubes_to_correct = cubes_to_correct

      sens_wvl = self.spectral_sensitivity.index
      sens_curve = np.array(self.spectral_sensitivity.spectral_sensitivity)
      for cubename in cubes_to_correct:
        
        cube_wvl = self.metadata[cubename]['wavelengths']
        if np.isin(cube_wvl, sens_wvl).mean() != 1:
          missing = set(cube_wvl).difference(sens_wvl)
          print(f"Attention! Cube '{cubename}' wavelengths are not in sensitivity data.")
          print(f"Missing wavelengths in sens. data: {missing}")
          print(f"Cube {cubename} not corrected by sensitivity data.")
          continue

        if not self.processed:
          print(f"Correcting by spectral sensitivity RAW cube '{cubename}'...")
          cube_corrected = self.raw[cubename] / sens_curve
        else:
          print(f"Correcting by spectral sensitivity PROCESSED cube '{cubename}'...")
          cube_corrected = self.processed[cubename] / sens_curve
        
        self.processed[cubename] = np.around(data_corrected, decimals = 2)

      print('Correction of cubes by spectral sensitivity done. See corrected cubes in attribute self.processed .\n--------------------------------')
    else: print('Attention! No data correction by spectral sensitivity took place.\n--------------------------------') 

#========== GET RGB ===============

  def get_rgb(self, cube_to_view, which_data='raw', color_bands=None):
  
    self.rgb_info = {}
    if color_bands is None:
      color_bands = self.color_bands['nuance']
    if self.data_source == 'goldeneye' or self.data_source == 'snapshot':
      color_bands = self.color_bands['goldeneye']
    
    data = getattr(self, which_data)
    cube = data[cube_to_view]

    red_bands = color_bands['red']
    green_bands = color_bands['green']
    blue_bands = color_bands['blue']

    # Extract data for each channel
    red_data = np.mean(cube[:, :, red_bands], axis=-1)
    green_data = np.mean(cube[:, :, green_bands], axis=-1)
    blue_data = np.mean(cube[:, :, blue_bands], axis=-1)

    # Normalize the data to [0, 1]
    normalized_red = (red_data - np.min(red_data)) / (np.max(red_data) - np.min(red_data))
    normalized_green = (green_data - np.min(green_data)) / (np.max(green_data) - np.min(green_data))
    normalized_blue = (blue_data - np.min(blue_data)) / (np.max(blue_data) - np.min(blue_data))

    # Stack the channels to create an RGB image
    self.rgb_array = np.stack([normalized_red, normalized_green, normalized_blue], axis=-1)
    self.rgb_image = Image.fromarray((self.rgb_array * 255).astype(np.uint8))
    self.rgb_info = {'time': self.time(), 'cubename': cube_to_view, 'which_data': which_data, 'color_bands': color_bands}
    return self

  #========= SAVE RGB ======================

  def save_rgb(self, filepath=None):
    if filepath is None:
      cubename = os.path.splitext(self.rgb_info['cubename'])[0]
      filename = f"RGB_Python_{self.sample}_{cubename}.png"
      filepath = os.path.join(os.path.dirname(self.data_path), filename)
    self.rgb_image.save(filepath)
    print(f"RGB saved at: {filepath}")

#=========== VIEW ======================
# x1=None, y1=None, width=None, height=None, color = 'red', title = None, fontsize = 12, filename = None, savefig = False, 
  
  def view(self, cube_to_view, which_data='raw', ax=None, **kwargs):
    
    self.get_rgb(cube_to_view, which_data, **kwargs)
    if ax is None:
      fig, ax = plt.subplots()
    else:
      ax = ax
    ax.imshow(self.rgb_array)
    self.ax = ax
    
    return self

#========= ROI ========================

  def reset_rois(self):
    self.rois = pd.DataFrame(columns=['coords', 'style', 'label'])
  
#---------------------------------------------

  def roi(self, coords=(0, 0, 0, 0), style='yyxx', edgecolor='red', linewidth = 0.7, linestyle='-', facecolor='none', keep=False, label=None, **kwargs):
    
    params = {
      'facecolor': facecolor,
      'linewidth': linewidth,
      'edgecolor': edgecolor,
      'linestyle': linestyle,
      **kwargs
    }
    if style == 'yyxx':
      y1, y2, x1, x2 = coords
      width = abs(x2-x1)
      height = abs(y2-y1)
    elif style == 'xyxy':
      x1, y1, x2, y2 = coords
      width = abs(x2-x1)
      height = abs(y2-y1)
    elif style == 'xywh':
      x1, y1, width, height = coords
      x2 = x1+width
      y2 = y1+height

    # causes strange behaviour when you don't expect
    y1 = min(y1, y2)
    y2 = max(y1, y2)
    x1 = min(x1, x2)
    x2 = max(x1, x2)
    
    rect = patches.Rectangle((x1, y1), width, height, **params)
    self.ax.add_patch(rect)

    self.selected_rows = slice(y1, y2)
    self.selected_cols = slice(x1, x2)

    roi = '_'.join(map(str, coords))
    style = style
    label = label
    roi_dict = {'coords': coords, 'style': style, 'label': label}
    self.last_roi = roi_dict
    if keep==True:
      self.rois.loc[roi] = roi_dict

    return self

#========= CROP ==================

  def crop(self, y1 = None, y2 = None, x1 = None, x2 = None):
    coords = [y1, y2, x1, x2]
    if all(coord is None for coord in coords):
      try:
        rows = self.selected_rows
        cols = self.selected_cols
      except:
        print("Oops! Seems you haven't specified any coordinates.")
    elif all(coords):
      rows = slice(min(y1, y2), max(y1, y2))
      cols = slice(min(x1, x2), max(x1, x2))
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

    if self.mask is not None:
      self.mask = self.mask[rows, cols]

#======== BINNING =====================

  def bin(self, bin_size):
    for cubename in self.raw.keys():
      cube = self.raw[cubename]
      cube_binned = bin_cube(cube, bin_size)
      self.raw[cubename] = cube_binned
      self.metadata[cubename]['num_rows'] = cube_binned.shape[0]
      self.metadata[cubename]['num_cols'] = cube_binned.shape[1]

    if self.processed:
      for cubename in self.processed.keys():
        cube = self.processed[cubename]
        cube_binned = bin_cube(cube, bin_size)
        self.processed[cubename] = cube_binned

    if self.normalized:
      for cubename in self.normalized.keys():
        cube = self.normalized[cubename]
        cube_binned = bin_cube(cube, bin_size)
        self.normalized[cubename] = cube_binned

    if self.mask is not None:
      self.mask = bin_mask(self.mask, bin_size)

#======== RESHAPE ====================
  
  def reshape(self, which_data = 'raw'):
    data_to_process = getattr(self, which_data)
    self.reshaped_input = which_data
    for cubename in data_to_process.keys():
      cube = data_to_process[cubename]
      cube_reshaped = np.reshape(cube, (cube.shape[0]*cube.shape[1], cube.shape[2]))
      self.reshaped[cubename] = cube_reshaped

#========== PCA ======================

  def get_pcs(self, cubes_to_analyse = None, components = 3, which_data = 'raw', df = False, subset_indices = None, extra_transform = False, trans_factor = 0.5, trans_inplace = False):
    
    data_to_process = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data_to_process.keys())
    
    for cubename in cube_names:
      cube = data_to_process[cubename]
      cube_reshaped = cube.reshape(cube.shape[0]*cube.shape[1], cube.shape[2]).astype(np.float64)
      if subset_indices is not None:
        cube_reshaped = cube_reshaped[subset_indices]
      pca = PCA(n_components = components)
      PCs = pca.fit_transform(cube_reshaped)
      
      if extra_transform == True:
        PCs_transformed = np.sign(PCs) * np.abs(PCs) ** trans_factor
        if df == True:
          columns = [f'PC{comp}' for comp in range(1, components+1)]
          PCs_transformed = pd.DataFrame(PCs, columns = columns)
        self.pcs_transformed[cubename] = PCs_transformed
      
      if trans_inplace == True:
        continue
      else:
        if df == True:
          columns = [f'PC{comp}' for comp in range(1, components+1)]
          PCs = pd.DataFrame(PCs, columns = columns)
        self.pcs[cubename] = PCs

    return self

  # def get_pcs_subset(self, ):
  #   data_to_process = getattr(self, which_data)
  #   if cubes_to_analyse:
  #     cube_names = cubes_to_analyse
  #   else:
  #     cube_names = list(data_to_process.keys())

#========= NORMALIZE ===============

  def normalize(self, cubes_to_analyse = None, which_data = 'raw', how = 'to_max'):
    data, cube_names = self.get_data(which_data, cubes_to_analyse)
    info = self.get_info(which_data)
    
    if how == 'to_max':
      for cubename in cube_names:
        cube = data[cubename]
        cube_max = np.max(cube, axis = 2, keepdims = True) + np.finfo(float).eps
        cube_normalized = cube / cube_max
        self.normalized[cubename] = cube_normalized
        self.normalized_info[cubename] = {'how': how,'wvls': info[cubename]['wvls']}
      self.log[self.time()] = {'normalize_to_max': {'which_data': which_data, 'cubes_to_analyse': cube_names}}
    
    elif how == 'minmax':
      for cubename in cube_names:
        cube = data[cubename]
        cube_normalized = (cube - cube.max(axis=2, keepdims=True)) / (cube.max(axis=2, keepdims=True)-cube.min(axis=2, keepdims=True) + np.finfo(float).eps)        
        self.normalized[cubename] = cube_normalized
        self.normalized_info[cubename] = {'how': how,'wvls': info[cubename]['wvls']}
      self.log[self.time()] = {f'normalize_{how}': {'which_data': which_data, 'cubes_to_analyse': cube_names}}

    elif how == 'snv': # Tested: https://colab.research.google.com/drive/1-x13RJ7qjf-PD-BaR3gzvTUdnr-Xqhax#scrollTo=_6j1pCY_jdfp&line=1&uniqifier=1
      for cubename in cube_names:
        cube = data[cubename]
        cube_avg = np.mean(cube, axis = 2, keepdims = True)
        cube_std = np.std(cube, axis = 2, keepdims = True)
        cube_std[cube_std == 0] = 1
        cube_snv = (cube - cube_avg) / cube_std
        self.normalized[cubename] = cube_snv
        self.normalized_info[cubename] = {'how': how,'wvls': info[cubename]['wvls']}
      self.log[self.time()] = {'SNV': {'which_data': which_data, 'cubes_to_analyse': cube_names}}

    elif how == 'zscale':
      for cubename in cube_names:
        cube = data[cubename]
        cube_avg = np.mean(cube, axis = (0, 1), keepdims = True)
        cube_std = np.std(cube, axis = (0, 1), keepdims = True)
        cube_std[cube_std == 0] = 1
        cube_zscaled = (cube - cube_avg) / cube_std
        self.normalized[cubename] = cube_zscaled
        self.normalized_info[cubename] = {'how': how,'wvls': info[cubename]['wvls']}
      self.log[self.time()] = {'ZScale': {'which_data': which_data, 'cubes_to_analyse': cube_names}}

    elif how == 'by_band':
      for cubename in cube_names:
        cube = data[cubename]
        cube_max = np.max(cube, axis = (0, 1), keepdims=True) + np.finfo(float).eps
        cube_normalized = cube / cube_max
        self.normalized[cubename] = cube_normalized
        self.normalized_info[cubename] = {'how': how,'wvls': info[cubename]['wvls']}
      self.log[self.time()] = {'normalize_by_band': {'which_data': which_data, 'cubes_to_analyse': cube_names}}

#======CORRECTION BY ANY ========
  
  def correct(self, by='exposure_time_ms', which_data=None, cubes_to_correct=None, correction_data: Optional[pd.Series] = None, normalized_to=None):
    # Date created: 2025-10-07
    if not which_data:
      if self.processed:
        which_data = 'processed'
      else:
        which_data = 'raw'
    
    data, cubenames = self.get_data(which_data, cubes_to_correct)
    info = self.get_info(which_data)
      
    if correction_data is None:
      correction_data = self.metadata_df[by]
    
    if normalized_to:
      factor = getattr(correction_data, normalized_to)()
      correction_data = correction_data / factor

    for cubename in cubenames:
      print(f"Correcting '{cubename}' by '{correction_data.name}' from '{which_data}' data...")
      value = correction_data[cubename.split('.')[0]]
      self.processed[cubename] = data[cubename] / value
      
      cube_info = {
        'processing': 'data_correction',
        'by': correction_data.name,
        'which_data': which_data,
        'cubes_to_correct': cubes_to_correct,
        'correction_data': correction_data,
        'wvls': info[cubename]['wvls']
      }
      self.processed_info[cubename] = cube_info
    
    return self

#======CORRECTION BY EXPOSURE TIME========
  
  def correct_by_exposure(self, cubes_to_analyse=None, which_data='raw', per_wavelength=False, exposure_data: Optional[pd.Series] = None):
    # Date created: 2025-05-23
    data, cube_names = self.get_data(which_data, cubes_to_analyse)
    info = self.get_info(which_data)
      
    if exposure_data is None:
      exposure_col = [col for col in self.metadata_df.columns if 'exposure_time' in col][0]
      exposure_data = self.metadata_df[exposure_col]

    for cubename in cube_names:
      exposure_val = exposure_data[cubename.split('.')[0]]
      self.processed[cubename] = data[cubename] / exposure_val
      
      cube_info = {
        'processing': 'correct_by_exposure',
        'which_data': which_data,
        'per_wavelength': per_wavelength,
        'exposure_data': exposure_data,
        'wvls': info[cubename]['wvls']
      }
      self.processed_info[cubename] = cube_info

#======= GAUSSIAN FILTER ===============

  def gaussian_filter(self, cubes_to_analyse=None, which_data='raw', sigma=(0, 0, 0)):
    
    self.processed_info = {}
    data = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = sorted(ensure_list(cubes_to_analyse))
    else:
      cube_names = sorted(ensure_list(data.keys()))

    for cubename in cube_names:
      self.processed[cubename] = gf(data[cubename], sigma=sigma)
      self.processed_info[cubename] = {'time': self.time(), 'source_data': which_data, 'processing': 'gaussian_filter', 'sigma': sigma}
    
    return self

#==== THRESHOLD BANDS ===========

  def threshold_bands(self, cubes_to_analyse, which_data='raw', quantile=0.995):
    
    self.thresholded = {}
    data = getattr(self, which_data)

    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data.keys())
    
    for cubename in cube_names:
      print(f"Thresholding cube '{cubename}' by band at q={quantile}")
      cube = data[cubename]
      band_quantiles = np.quantile(cube, quantile, axis=(0, 1), keepdims=True)
      cube_clipped = np.minimum(cube, band_quantiles)
      self.thresholded[cubename] = cube_clipped

#========== Z-SCALE 2D DATA ============================

  def scale(self, labels: list = None, which_data = 'pcs', how = 'mean'):
  
    self.scaled_data = {}
    data_dict = getattr(self, which_data)
    if labels:
      labels = ensure_list(labels)
    else:
      labels = list(data_dict.keys())

    for label in labels:
      data = data_dict[label]
      if how == 'mean':
        data_stat = data.mean(axis = 0)
      elif how == 'median':
        data_stat = np.median(data, axis = 0)
      data = (data - data_stat) / data.std(axis=0)
      
      self.scaled_data[label] = data

#============= MASK ===============

  def read_mask(self, filepath, mask_labels = None):
    img = Image.open(filepath)
    img_arr = np.array(img)
    
    self.mask = img_arr
    self.mask_labels = mask_labels
    self.pixel_labels = img_arr.reshape(img_arr.shape[0]*img_arr.shape[1])
    print(f"Values in Mask: {np.unique(img_arr)}")
    
    if mask_labels is not None:
      self.pixel_labels_str = [mask_labels[num_lab] for num_lab in self.pixel_labels]

    print(f"Assigned Labels (string labels to mask numeric values - not sure, need to check): {mask_labels}")
    # if show=True:
    #   plt.imshow(img_arr);
    return self

  def save_mask(self, filepath, format = None):
    
    if format:
      filepath = os.path.splitext(filepath)[0] + '.' + format.lower()

    # Ensure the mask data is in a format suitable for images
    if self.mask.dtype != np.uint8:
      raise ValueError("The mask array must have a dtype of 'uint8'.")

    try:
      # Convert the mask to a PIL Image and save
      img = Image.fromarray(self.mask)
      img.save(filepath, format = format)
      print(f"Mask saved successfully at {filepath}")
    except Exception as e:
        print(f"An error occurred while saving the mask: {e}")

#======= GET WVLS ===============
  
  def get_wvls(self, cubename):
    cubename = cubename.split('.')[0]
    emission_start = int(self.metadata_df.loc[cubename, 'emission_start_nm'])
    emission_end = int(self.metadata_df.loc[cubename, 'emission_end_nm'])
    step = int(self.metadata_df.loc[cubename, 'step_nm'])
    wvls = np.arange(emission_start, emission_end+1, step)
    return wvls

#=======GET SPECTRA============

  def get_spectra_v1(self, cubes_to_analyse, which_data='raw', mask_label=None, df=False, wvls=False, long=False, label=None, sample_size=None):

    self.spectra = {}
    self.spectra_info = {}
    
    # Maybe no need for estimators, quite easy to do outside
    # estimators = {
    #     'mean': np.mean,
    #     'median': np.median
    # }  
    
    data_to_process = getattr(self, which_data)

    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data_to_process.keys())

    if mask_label is not None:
      self.where = np.where(self.mask == mask_label)
      for cubename in cube_names:
        cube = data_to_process[cubename]
        segment = cube[self.where]
        self.spectra[cubename] = segment

        self.spectra_info[cubename] = {}
        self.spectra_info[cubename]['label'] = label
        self.spectra_info[cubename]['mask_label'] = mask_label
        if wvls == True:
          if which_data == 'combined':
            self.spectra_info[cubename]['wvls'] = self.combined_metadata[cubename]['wavelengths']
          else:
            self.spectra_info[cubename]['wvls'] = self.get_wvls(cubename)
    else:
      rows = self.selected_rows
      cols = self.selected_cols
      for cubename in cube_names:
        cube = data_to_process[cubename]
        segment = cube[rows, cols]
        segment = segment.reshape(segment.shape[0]*segment.shape[1], segment.shape[-1])
        self.spectra[cubename] = segment

        self.spectra_info[cubename] = {}
        self.spectra_info[cubename]['label'] = label
        self.spectra_info[cubename]['rows'] = (rows.start, rows.stop)
        self.spectra_info[cubename]['cols'] = (cols.start, cols.stop)
        if wvls == True:
          if which_data == 'combined':
            self.spectra_info[cubename]['wvls'] = self.combined_metadata[cubename]['wavelengths']
          else:
            self.spectra_info[cubename]['wvls'] = self.get_wvls(cubename)

    if sample_size is not None:
      np.random.seed(42)
      for cubename in cube_names:
        random_index = np.random.choice(self.spectra[cubename].shape[0], sample_size, replace=False)
        self.spectra[cubename] = self.spectra[cubename][random_index]

    # if estimator in estimators:
    #   for cubename in cube_names:
    #     self.spectra[cubename] = estimators[estimator](self.spectra[cubename], axis=0)
    # else:
    #   raise ValueError(f"Unknown estimator: {estimator}")

    if df == True:
      for cubename in cube_names:
        if wvls == True:
          df = pd.DataFrame(self.spectra[cubename], columns = self.spectra_info[cubename]['wvls'])
        else:
          df = pd.DataFrame(self.spectra[cubename])
        if long == True:
          df.insert(0, 'Label', label)
          df = df.melt(id_vars='Label', var_name='Wavelength', value_name='Intensity')
        self.spectra[cubename] = df

    return self

#======= GET INDEX =============
  
  def get_index(self, coords = None, roi_name = None, mask_value=None, **kwargs):
    if roi_name is None and mask_value is None and coords is None:
      y1, y2 = self.selected_rows.start, self.selected_rows.stop
      x1, x2 = self.selected_cols.start, self.selected_cols.stop
      coords = (y1, y2, x1, x2)
      where = coords_to_where(self.size, coords)
    elif coords and roi_name is None and mask_value is None:
      where = coords_to_where(self.size, coords)
    elif roi_name and coords is None and mask_value is None:
      coords = self.rois.loc[roi_name, 'coords']
      where = coords_to_where(self.size, coords)
    elif mask_value is not None and coords == None and roi_name is None:
      where = np.where(self.mask == mask_value)
    return where, coords, roi_name, mask_value

#========== EEM ===============

  def get_eem(self, cubes_to_analyse=None, which_data='raw', coords=None, roi_name=None, mask_value=None):
    data, cube_names = self.get_data(which_data, cubes_to_analyse)
    
    params = locals()
    params.pop('self', None)
    where, coords, roi_name, mask_value = self.get_index(**params)

    spectra = []
    for cubename in cube_names:
      wvls = self.get_wvls(cubename)
      cube_segment = data[cubename][where]
      spectrum = np.mean(cube_segment, axis = 0)
      spectrum_df = pd.DataFrame([spectrum], index=[cubename.split('.')[0]], columns=wvls)
      spectra.append(spectrum_df)
    self.eem = pd.concat(spectra).sort_index(axis=1).sort_index(axis=0, ascending=False)
    
    self.eem_info = {
      'coords': coords,
      'roi_name': roi_name,
      'mask_value': mask_value,
      'which_data': which_data,
      'cubenames': cube_names,
      }
    
    return self

#------------------------------------

  def quick_eem(self, cubes_to_analyse = None, which_data = 'raw', mask_value = None, transform = False, plot = True, vmin = None, vmax = None, axis_ratio = None, title = None, region = None, ax = None, cbar_ax = None, fontsize = 'medium', ticksize = 'medium', xtickstep = 2, also_spectra = True):
      
    data_to_process = getattr(self, which_data)
    
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = self.names

    if mask_value:
      where = np.where(self.mask == mask_value)
    else:
      rows = self.selected_rows
      cols = self.selected_cols
    
    eem = pd.DataFrame()
    self.spectra_combined_avg = pd.DataFrame()
    for cubename in cube_names:
      if mask_value:
        cube_segment = data_to_process[cubename][where]
        cube_segment_avg = np.mean(cube_segment, axis = 0).reshape(1, cube_segment.shape[-1])
      else:
        cube_segment = data_to_process[cubename][rows, cols, :]
        cube_segment_avg = np.mean(cube_segment, axis = (0, 1)).reshape(1, cube_segment.shape[-1])
      ex = pd.Series(str(self.metadata[cubename]['ex']))
      wavelengths = self.metadata[cubename]['wavelengths']
      spectrum_avg = pd.DataFrame(cube_segment_avg, columns = wavelengths, index = ex)
      eem = pd.concat([eem, spectrum_avg], axis = 0)
    if also_spectra == True:
      self.spectra_avg = eem.sort_index(ascending=True).T
      self.spectra_combined_avg = eem.stack().reset_index()
      self.spectra_combined_avg.columns = ['ex', 'em', 'spectrum']
    eem = eem.sort_index(ascending = False)

    if transform == True:
      eem_stacked = eem.stack()
      eem_average = eem_stacked.mean()
      eem_std = eem_stacked.std()
      eem_zscaled = ((eem_stacked - eem_average) / eem_std).unstack()
      eem = eem_zscaled.applymap(np.exp)

    self.last_eem = eem
    
    if plot == True:
      if ax is None:
        fig, ax = plt.subplots()
      else:
        ax = ax
      sns.heatmap(eem, cmap = 'coolwarm', ax = ax, vmin = vmin, vmax = vmax)
      if title == None:
        if region == None:
          this_region = f"Segment mask='{mask_value}'" if mask_value else f"Y={rows.start}:{rows.stop}, X={cols.start}:{cols.stop}"
        else:
          this_region = f"Segment mask='{mask_value}'" if mask_value else f"{region.capitalize()}: Y={rows.start}:{rows.stop}, X={cols.start}:{cols.stop}"
        title = f"Average EEM\n({this_region}, {which_data.capitalize()} Data)"
      ax.set_title(title, size = fontsize)
      ax.set_xlabel('Emission', size=fontsize)
      ax.set_ylabel('Excitation', size=fontsize)
      ax.tick_params(axis='x', rotation=45, labelsize=ticksize)
      ax.tick_params(axis='y', rotation=0, labelsize=ticksize)

#========== GET SPECTRA NEW ================

  def get_spectra(self, cubes_to_analyse=None, which_data='raw', coords=None, roi_name=None, mask_value=None, sample=None):
    data, cube_names = self.get_data(which_data, cubes_to_analyse)
    
    params = locals()
    params.pop('self', None)
    where, coords, roi_name, mask_value = self.get_index(**params)

    spectra_dfs = []
    for cubename in cube_names:
      cube = data[cubename]
      info = getattr(self, which_data + '_info')
      wvls = info[cubename]['wvls']

      spectra = cube[where]
      spectra_df = pd.DataFrame(spectra, columns=wvls)
      spectra_df['cubename'] = cubename.split('.')[0]
      spectra_df.set_index('cubename', inplace=True)
      if sample:
        spectra_df = spectra_df.sample(sample)
      spectra_dfs.append(spectra_df)
    self.spectra = pd.concat(spectra_dfs).sort_index(axis=1)
    spectra_info = {
      'which_data': which_data,
      'coords': [coords] * len(self.spectra),
      'coords_style': 'yyxx',
      'roi_name': roi_name,
      'mask_value': mask_value,
    }
    self.spectra_ids = []
    for i, item in enumerate(spectra_info.items()):
      self.spectra.insert(i, item[0], item[1])
      self.spectra_ids.append(item[0])
    self.spectra_wvls = sorted(list(set(self.spectra.columns).difference(set(self.spectra_ids))))
    return self

#----------------------------------------------------

  def spectra_from_rois(self, cubes_to_analyse=None, which_data='raw', sample=None):
    rois = self.rois.index.tolist()
    spectra = []
    for roi in rois:
      self.get_spectra(cubes_to_analyse, which_data, roi_name=roi, sample=sample)
      spectra.append(self.spectra)
    self.spectra = pd.concat(spectra)
    return self
  
#----------------------------------------------------

  def spectra_to_long(self):
    self.spectra = self.spectra.melt(self.spectra_ids, self.spectra_wvls, 'wavelength', 'intensity')

#============ COMBINE =====================

  def combine(self, cubes_to_analyse = None, which_data = 'raw', label = None):
    data, cube_names = self.get_data(which_data, cubes_to_analyse)
    
    if label == None:
      excitations = [cubename.split("_")[0].split(".")[0] for cubename in cube_names]
      label = f"{which_data.capitalize()}_{'_'.join(excitations)}"

    # self.combined[label] = None
    # self.combined_wvls[label] = np.empty((0), dtype = np.int64)
    self.combined_metadata[label] = {}
    self.combined_metadata[label]['wavelengths'] = np.empty((0), dtype = np.int64)
    self.combined_info[label] = {'wvls': []}

    if label in self.combined:
      del self.combined[label]

    for cubename in cube_names:
      print(f"Combining '{cubename}' from '{which_data}' data...")
      cube = data[cubename]
      wavelengths = self.metadata[cubename]['wavelengths'] # check usage and delete
      wavelengths = wavelengths if wavelengths is not None else np.array([]) # check usage and delete
      wvls = self.get_info(which_data)[cubename]['wvls'] # new 2025-05-25 
      wvls = [f"{cubename.split('.')[0]}_{wvl}" for wvl in wvls] # new 2025-05-25
      if label not in self.combined:
        self.combined[label] = np.empty((cube.shape[0], cube.shape[1], 0), dtype = np.float32)
      self.combined[label] = np.concatenate((self.combined[label], cube), axis = 2) # check usage and delete
      self.combined_metadata[label]['wavelengths'] = np.concatenate((self.combined_metadata[label]['wavelengths'], wavelengths)) # check usage and delete
      self.combined_info[label]['wvls'] += wvls # new 2025-05-25
    self.combined_metadata[label]['source'] = which_data # check usage and delete
    self.combined_metadata[label]['cubes'] = cube_names # check usage and delete
    self.combined_info[label]['source'] = which_data # new 2025-05-25
    self.combined_info[label]['cubenames'] = cube_names # new 2025-05-25

#============ AVERAGE ======================
  
  def average(self, cubes_to_analyse = None, which_data = 'raw', description = None):
    data = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data.keys())

    if description == None:
      excitations = [cubename.split("_")[0].split(".")[0] for cubename in cube_names]
      description = f"{which_data.capitalize()}_{Avg}_{'_'.join(excitations)}"

    cubes_list = []
    for cubename in cube_names:
      cube = data[cubename]
      cubes_list.append(cube)
      
    hypercube = np.stack(cubes_list, axis = 0)
    cubes_average = np.mean(hypercube, axis = 0)
    
    self.averaged[description] = cubes_average
    self.metadata[description] = {'wavelengths':self.metadata[self.names[0]]['wavelengths']}
    print(f"Cubes averaged: {cube_names}")
    print(f"Find averaged data (label = '{description}') in cubes.averaged attribute.")

#============= SUM =======================
  
  def sum(self, cubes_to_analyse = None, which_data = 'raw', description = None):
    data = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data.keys())

    if description == None:
      excitations = [cubename.split("_")[0].split(".")[0] for cubename in cube_names]
      description = f"{which_data.capitalize()}_{Sum}_{'_'.join(excitations)}"

    cubes_list = []
    for cubename in cube_names:
      cube = data[cubename]
      cubes_list.append(cube)
      
    hypercube = np.stack(cubes_list, axis = 0)
    cubes_sum = np.sum(hypercube, axis = 0)
    
    self.summed[description] = cubes_sum
    self.metadata[description] = {'wavelengths':self.metadata[self.names[0]]['wavelengths']}
    print(f"Cubes summed: {cube_names}")
    print(f"Find summed data (label = '{description}') in cubes.summed attribute.")

#============= SAVE TIFF =================

  def save_tiff(self, cubes_to_save = None, which_data = 'raw', mode = 'cubes', destination = None, description = None):
    
    data = getattr(self, which_data)
    
    if cubes_to_save:
      cube_names = ensure_list(cubes_to_save)
    else:
      cube_names = list(data.keys())

    if destination == None:
      destination = os.path.dirname(self.data_path)
    if description == None:
      cubenames_bases = [str.split(cubename, '.')[0] for cubename in cube_names]
      description = f"Tiff_{mode.capitalize()}_{which_data.capitalize()}_{'_'.join(cubenames_bases)}"
    output_path = os.path.join(destination, description)
    os.makedirs(output_path, exist_ok=True)

    if mode == 'cubes':
      for cubename in cube_names:
        cube = data[cubename].astype(np.uint16)
        cube_for_tiff = cube.transpose(2, 0, 1)
        cubename_base = str.split(cubename, '.')[0]
        cube_path = os.path.join(output_path, f'{cubename_base}.tif')
        tiff.imwrite(cube_path, cube_for_tiff)
    elif mode == 'slices':
      for cubename in cube_names:
        cubename_base = str.split(cubename, '.')[0]
        cube_path = os.path.join(output_path, cubename_base)
        os.makedirs(cube_path, exist_ok = True)

        cube = data[cubename]
        bands_num = cube.shape[2]
        for band in range(bands_num):
          band_data = cube[:, :, band]
          if which_data != 'combined':
            wavelengths = self.metadata[cubename]['wavelengths']
            wvl = wavelengths[band]
          else: wvl = band
          slicename = f'{cubename_base}_{wvl:04}.tif'
          slicepath = os.path.join(cube_path, slicename)
          img = Image.fromarray(band_data)
          print(f"Saving band {band} to {slicepath}...")
          img.save(slicepath, format = "TIFF")

  def print_log(self, indent = None):
    print('Attention! Not all functions have been connected to the log. This is a feature under development.')
    print(json.dumps(self.log, indent = indent))

  def time(self):
    yerevantime = pytz.timezone('Asia/Yerevan')
    return datetime.now().astimezone(yerevantime).strftime('%y%m%d_%H%M%S')

  def get_data(self, which_data, cubes_to_analyse=None):
    data = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = sorted(ensure_list(cubes_to_analyse))
    else:
      cube_names = sorted(ensure_list(data.keys()))
    return data, sorted(cube_names)
  
  def get_info(self, which_data):
    info = getattr(self, f"{which_data}_info")
    return info
######################################

def read_cube_slices(data_path, cubename):
  from pathlib import Path
  cube_path = Path(data_path, cubename)
  bands = cube_path.glob('*.tif*')
  cube = []
  for bandpath in sorted(list(bands)):
    # band = Image.open(bandpath)
    print(f"reading {bandpath.name}")
    band = tiff.imread(bandpath) #.transpose(1, 2, 0) 
    band = np.array(band, dtype=np.float32)
    cube.append(band)
  cube = np.stack(cube, axis=-1)
  return cube

def read_metadata(metadata_path, sample_id=None):
  metadata = pd.read_csv(metadata_path)
  metadata.columns = [col.lower() for col in metadata.columns]
  if sample_id:
    metadata = metadata[metadata['sample_id'] == sample_id]
  metadata['cube_name'] = metadata.cube_name.astype(str)
  metadata.set_index('cube_name', inplace = True)
  return metadata

# deleted from utils.py
def bin_cube(cube, bin_size):
  rows, cols, bands = cube.shape
  if rows % bin_size != 0 or cols % bin_size != 0:
    print("Attention! One of cube dimensions is not divisible by bin_size. It will be cropped to fit the correct dimensions.")
    rows = rows - rows % bin_size
    cols = cols - cols % bin_size
    cube = cube[:rows, :cols, :]

  new_rows = rows // bin_size
  new_cols = cols // bin_size

  cube_binned = cube.reshape(new_rows, bin_size, new_cols, bin_size, bands).mean(axis = (1, 3))

  return cube_binned

#========== GET WHERE =======================
def coords_to_where(mask_size, coords, style='yyxx'):
  y1, y2, x1, x2 = coords
  mask = np.zeros(mask_size, dtype=bool)
  mask[y1:y2, x1:x2] = True
  indices = np.where(mask)
  return indices

