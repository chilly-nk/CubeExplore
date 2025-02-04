import os
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pytz
import json
import tifffile as tiff
import spectral as spy
import spectral.io.envi as envi
from datetime import datetime
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import zoom

"""Initiate pyimagej (at fiji mode)"""
# import imagej
# ij = imagej.init('sc.fiji:fiji')

class Cubes:
  def __init__(self, data_path, metadata_path = None, cubes_to_load = None, data_source = 'nuance'):
    
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
    
    self.cubes_to_analyse = None
    
    self.raw = {}
    self.metadata = {}
    self.tls_spectrum = None
    self.spectral_sensitivity = None
    self.processed = {}
    self.normalized = {}
    
    self.combined = {}
    # self.combined_wvls = {} # maybe delete
    self.combined_metadata = {}
    
    self.reshaped = {}

    self.averaged = {}
    self.summed = {}

    self.selected_rows = None
    self.selected_cols = None
    
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
      self.read_metadata(metadata_path)
    
    if cubes_to_load:
      cube_names = sorted(cubes_to_load)
    else:
      cube_names = sorted(os.listdir(data_path))
    self.names = cube_names
    
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
      cube = np.array(img_loaded, dtype = np.float32)
      
      self.raw[cubename] = cube
      
      ex = cubename.split("_")[0].split(".")[0]
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
        self.metadata[cubename]['notes'] = self.metadata_df.loc[ex, 'notes']
        self.metadata[cubename]['wavelengths'] = np.array(range(emission_start, emission_end+1, step))

  def read_metadata(self, metadata_path):
    metadata = pd.read_csv(metadata_path)
    metadata['cube_name'] = metadata.cube_name.astype(str)
    metadata.set_index('cube_name', inplace = True)
    self.metadata_df = metadata

  def read_tls_data(self, tls_spectrum_path):
    # Load Correction Data (TLS Basic Wavelength Scan, several scans repetitions). All scans must have the same start, stop, step
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

#=========== VIEW ======================

  def view(self, cube_to_view: str, y1 = None, y2 = None, x1 = None, x2 = None, blue_bands = range(3, 9), green_bands = range(13, 19), red_bands = range(23, 29), ax = None, color = 'red', pic_only = False, title = None, fontsize = 12, filename = None, savefig = False):
    
    cube = self.raw[cube_to_view]
    if filename:
      filename = filename+'.png'
    elif title:
      filename = title+'.png'
    else:
      filename = f"view_{self.metadata[cube_to_view]['ex']}.png"
    
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
    self.rgb = rgb_image

    if pic_only == True:
      rgb_image_norm = (rgb_image * 255).astype(np.uint8)
      image = Image.fromarray(rgb_image_norm)
      image.save(f"Pic_Only_{filename}")
    else:
      # Display the RGB image
      if ax is None:
        fig, ax = plt.subplots()
      else:
        ax = ax
      ax.imshow(rgb_image);

      if title:
        ax.set_title(title, size = fontsize)
    
      # Set the boundaries
      coords = pd.Series([y1, y2, x1, x2])
      if coords.notna().all():
        color = color
        ax.axvline(x = x1, color = color, linewidth = 0.7, linestyle = '--');
        ax.axvline(x = x2, color = color, linewidth = 0.7, linestyle = '--');
        ax.axhline(y = y1, color = color, linewidth = 0.7, linestyle = '--');
        ax.axhline(y = y2, color = color, linewidth = 0.7, linestyle = '--');
    
        self.selected_rows = slice(min(y1, y2), max(y1, y2))
        self.selected_cols = slice(min(x1, x2), max(x1, x2))
    
      if savefig == True:
        plt.savefig(filename, bbox_inches = 'tight', dpi = 200)
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
    
    # plt.show()

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

  def get_pcs(self, cubes_to_analyse = None, components = 3, which_data = 'raw', df = False, mask_array = None, extra_transform = False, trans_factor = 0.5, trans_inplace = False):
    
    data_to_process = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data_to_process.keys())
    
    for cubename in cube_names:
      cube = data_to_process[cubename]
      cube_reshaped = cube.reshape(cube.shape[0]*cube.shape[1], cube.shape[2]).astype(np.float64)
      if mask_array is not None:
        cube_reshaped = cube_reshaped[mask_array]
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

  # def get_pcs_subset(self, ):
  #   data_to_process = getattr(self, which_data)
  #   if cubes_to_analyse:
  #     cube_names = cubes_to_analyse
  #   else:
  #     cube_names = list(data_to_process.keys())

#========= NORMALIZE ===============

  def normalize(self, cubes_to_analyse = None, which_data = 'raw', how = 'to_max'):
    
    data = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data.keys())
    
    if how == 'to_max':
      for cubename in cube_names:
        cube = data[cubename]
        cube_max = np.max(cube, axis = 2, keepdims = True) + np.finfo(float).eps
        cube_normalized = cube / cube_max
        self.normalized[cubename] = cube_normalized
      self.log[self.time()] = {'normalize_to_max': {'which_data': which_data, 'cubes_to_analyse': cube_names}}
    
    elif how == 'snv': # Tested: https://colab.research.google.com/drive/1-x13RJ7qjf-PD-BaR3gzvTUdnr-Xqhax#scrollTo=_6j1pCY_jdfp&line=1&uniqifier=1
      for cubename in cube_names:
        cube = data[cubename]
        cube_avg = np.mean(cube, axis = 2, keepdims = True)
        cube_std = np.std(cube, axis = 2, keepdims = True)
        cube_snv = (cube - cube_avg) / cube_std
        self.normalized[cubename] = cube_snv
      self.log[self.time()] = {'SNV': {'which_data': which_data, 'cubes_to_analyse': cube_names}}

    elif how == 'zscale':
      for cubename in cube_names:
        cube = data[cubename]
        cube_avg = np.mean(cube, axis = (0, 1), keepdims = True)
        cube_std = np.std(cube, axis = (0, 1), keepdims = True)
        cube_zscaled = (cube - cube_avg) / cube_std
        self.normalized[cubename] = cube_zscaled
      self.log[self.time()] = {'ZScale': {'which_data': which_data, 'cubes_to_analyse': cube_names}}

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
    # This part must be eliminated after our masks are exact-value ones
    # img_arr[(img_arr < 50)] = 0
    # img_arr[(img_arr >= 50) & (img_arr < 125)] = 1
    # img_arr[(img_arr >= 125)] = 2
    
    self.mask = img_arr
    self.mask_labels = mask_labels
    print(f"Values in Mask: {np.unique(img_arr)}")
    print(f"Assigned Labels: {mask_labels}")
    plt.imshow(img_arr);

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
  

#========== EEM ===============

  def get_eem(self, cubes_to_analyse = None, which_data = 'raw', mask_label = None, transform = False, plot = True, vmin = None, vmax = None, axis_ratio = None, title = None, region = None, ax = None, cbar_ax = None, fontsize = 'medium', ticksize = 'medium', xtickstep = 2, also_spectra = True):
      
    data_to_process = getattr(self, which_data)
    
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = self.names

    if mask_label:
      where = np.where(self.mask == self.mask_labels[mask_label])
    else:
      rows = self.selected_rows
      cols = self.selected_cols
    
    eem = pd.DataFrame()
    self.spectra_combined_avg = pd.DataFrame()
    for cubename in cube_names:
      if mask_label:
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
          this_region = f"Segment '{mask_label}'" if mask_label else f"Y={rows.start}:{rows.stop}, X={cols.start}:{cols.stop}"
        else:
          this_region = f"Segment '{mask_label}'" if mask_label else f"{region.capitalize()}: Y={rows.start}:{rows.stop}, X={cols.start}:{cols.stop}"
        title = f"Average EEM\n({this_region}, {which_data.capitalize()} Data)"
      ax.set_title(title, size = fontsize)
      ax.set_xlabel('Emission', size=fontsize)
      ax.set_ylabel('Excitation', size=fontsize)
      ax.tick_params(axis='x', rotation=45, labelsize=ticksize)
      ax.tick_params(axis='y', rotation=0, labelsize=ticksize)

#============ COMBINE =====================

  def combine(self, cubes_to_analyse = None, which_data = 'raw', description = None):
    data = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = ensure_list(cubes_to_analyse)
    else:
      cube_names = list(data.keys())

    if description == None:
      excitations = [cubename.split("_")[0].split(".")[0] for cubename in cube_names]
      description = f"{which_data.capitalize()}_{'_'.join(excitations)}"

    # self.combined[description] = None
    # self.combined_wvls[description] = np.empty((0), dtype = np.int64)
    self.combined_metadata[description] = {}
    self.combined_metadata[description]['wavelengths'] = np.empty((0), dtype = np.int64)

    if description in self.combined:
      del self.combined[description]

    for cubename in cube_names:
      print(f"Combining '{cubename}' from '{which_data}' data")
      cube = data[cubename]
      wavelengths = self.metadata[cubename]['wavelengths']
      wavelengths = wavelengths if wavelengths is not None else np.array([])
      if description not in self.combined:
        self.combined[description] = np.empty((cube.shape[0], cube.shape[1], 0), dtype = np.float32)
      self.combined[description] = np.concatenate((self.combined[description], cube), axis = 2)
      self.combined_metadata[description]['wavelengths'] = np.concatenate((self.combined_metadata[description]['wavelengths'], wavelengths))
    self.combined_metadata[description]['source'] = which_data
    self.combined_metadata[description]['cubes'] = cube_names

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

def read_spectral_library(library_path):
  components = [f'C{i}' for i in range(1, 11)]
  spectral_library = pd.read_csv(library_path, sep = '\t', skiprows = 1)
  spectral_library = spectral_library.iloc[:, 1:12]
  spectral_library = spectral_library.T.reset_index().T
  spectral_library.columns = ['wavelength'] + components
  spectral_library = spectral_library.reset_index(drop = True)
  spectral_library = spectral_library.astype(float)
  return spectral_library

#===== NUANCE COMPONENTS ======================

class Components:
  def __init__(self, unmixing_path, n_components = 3, source = 'component_images'):
    
    self.unmixing_path = unmixing_path

    filenames = os.listdir(unmixing_path)
    if source == 'component_images':
      substrings = [f'C{component}' for component in range(1, n_components+1)]
      component_files = [filename for filename in filenames if any((sub in filename) and ('Data' not in filename) for sub in substrings)]
    elif source == 'component_data':
      substrings = [f'C{component}_Data' for component in range(1, n_components+1)]
      component_files = [filename for filename in filenames if any(sub in filename for sub in substrings)]
    self.filenames = component_files
    
    imar_list = []
    for component in sorted(component_files):
      component_path = os.path.join(unmixing_path, component)
      img = Image.open(component_path)
      imar = np.atleast_3d(np.array(img))[:, :, 0]
      imar_list.append(imar)
    self.stack = np.stack(imar_list, axis = 2)

  def get_mask(self, component_values: dict = {1:200, 2:100, 3:0}, threshold_quantiles = None):
    
    if threshold_quantiles:
      upper = max(threshold_quantiles)
      lower = min(threshold_quantiles)
      quantiles_upper = np.quantile(self.stack, upper, axis = (0, 1), keepdims = True)
      quantiles_lower = np.quantile(self.stack, lower, axis = (0, 1), keepdims = True)
      stack_thresholded = (self.stack < quantiles_upper) & (self.stack > quantiles_lower)
      stack_thresholded = stack_thresholded.astype(int)
      for component, value in component_values.items():
        i = component-1
        stack_thresholded[:, :, i][stack_thresholded[:, :, i] == 1] = value
        stack_thresholded[:, :, i][stack_thresholded[:, :, i] == 0] = 0
      self.mask = np.sum(stack_thresholded, axis = -1)
      return self

    mask = np.argmax(self.stack, axis = 2)
    for component, value in component_values.items():
      mask[mask == component-1] = value
    self.mask = mask
    return self

  def normalize(self, inplace=True):
    max_by_slice = self.stack.max(axis = (0, 1), keepdims = True)
    if inplace:
      self.stack = self.stack / max_by_slice
      return self
    else:
      stack_normalized = self.stack / max_by_slice
      return stack_normalized

  def means_by_mask(self, ref_mask, component_values: dict = {1:200, 2:100, 3:0}):
    self.means = {}
    for component, value in component_values.items():
      where = np.where(ref_mask == value)
      for i in range(len(component_values)):
        component_segment_mean = self.stack[:, :, i][where].mean()
        self.means[f"SegmC{component}_MeanC{i+1}"] = component_segment_mean
    print(f"Component means calculated per component per reference mask segment.")
    
  def save_rgb(self, filename = None, unmixing_path = None):
    if unmixing_path is None:
      unmixing_path = self.unmixing_path
    if filename is None:
      filename = f'PyRGB'
    filepath = os.path.join(unmixing_path, filename+'.png')

    data = (self.normalize(inplace=False) * 255).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(filepath, format = 'PNG')
    print(f"Pseudo-RGB saved successfully at {filepath}")

  def save_mask(self, filename = None, unmixing_path = None):
    if unmixing_path is None:
      unmixing_path = self.unmixing_path
    if filename is None:
      filename = f'PyMask'
    filepath = os.path.join(unmixing_path, filename+'.png')

    data = self.mask.astype(np.uint8)
    img = Image.fromarray(data)
    img.save(filepath, format = 'PNG')
    print(f'Mask saved successfully at {filepath}')
    return self



def get_nuance_mask(unmixing_path, n_components = 3, source = 'component_data', component_values: dict = {1:200, 2:100, 3:0}, composite = True, normalize = False, threshold_quantiles = None):
  
  filenames = os.listdir(unmixing_path)
  if source == 'component_image':
    substrings = [f'C{component}' for component in range(1, n_components+1)]
    component_files = [filename for filename in filenames if any((sub in filename) and ('Data' not in filename) for sub in substrings)]
  elif source == 'component_data':
    substrings = [f'C{component}_Data' for component in range(1, n_components+1)]
    component_files = [filename for filename in filenames if any(sub in filename for sub in substrings)]

  imar_list = []
  for component in sorted(component_files):
    component_path = os.path.join(unmixing_path, component)
    img = Image.open(component_path)
    imar = np.atleast_3d(np.array(img))[:, :, 0]
    imar_list.append(imar)
  my_composite = np.stack(imar_list, axis = 2)
  argmax_mask = np.argmax(my_composite, axis = 2)
  
  if normalize == True:
    slice_max = my_composite.max(axis = (0, 1), keepdims = True)
    my_composite = my_composite / slice_max
    argmax_mask = np.argmax(my_composite, axis = 2)
  

  for component, value in component_values.items():
    argmax_mask[argmax_mask == component-1] = value

  if composite == True:
    return argmax_mask, my_composite
  else:
    return argmax_mask

def read_mask(mask_path, new_values: dict = None, silent = True):
  img = Image.open(mask_path)
  mask = np.array(img)
  if silent == False:
    print(f"Original Values: {np.unique(mask)}")
  
  if new_values is not None:
    for old_value, new_value in new_values.items():
      mask[mask == old_value] = new_value
    if silent == False:
      print(f"New Values: {np.unique(mask)}")
  return mask

def ensure_list(input_value):
  if isinstance(input_value, str):
    return [input_value]
  elif isinstance(input_value, list):
    return input_value
  else:
    raise TypeError("Input must be either a string or a list")

  # def read_mask(self, filepath, mask_labels = None):  

  # This doesn't work yet
  # def savefile(self, name = 'cubes', path = str):
  #   filepath = os.path.join(path, name + '.pkl')
  #   with open(filepath, 'wb') as file:
  #     pickle.dump(self, file)

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

def bin_mask(mask, bin_size):
  rows, cols = mask.shape
  if rows % bin_size != 0 or cols % bin_size != 0:
    rows = rows - rows % bin_size
    cols = cols - cols % bin_size
    mask = mask[:rows, :cols]

  new_rows = rows // bin_size
  new_cols = cols // bin_size
  
  zoom_factor = 1 / bin_size
  return zoom(mask, zoom_factor, order=0)

