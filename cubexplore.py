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

# # Initiate pyimagej (at fiji mode)
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
    self.correction_data_ls = None
    self.spectral_sensitivity = None
    self.processed = {}
    self.normalized = {}
    
    self.combined = {}
    self.combined_wvls = {}
    self.combined_metadata = {}

    self.selected_rows = None
    self.selected_cols = None
    
    self.pcs = {}
    self.pcs_transformed = {}

    self.mask = None
    self.mask_labels = {}

    self.spectra = {} # By Cube
    self.spectra_avg = {} # By Cube
    self.spectra_combined = None # In progress
    self.spectra_combined_avg = None

    if metadata_path:
      self.get_metadata(metadata_path)
    
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
      # metadata = pd.read_csv(metadata_path)
      # metadata['excitation'] = metadata.excitation.astype(str)
      # metadata.set_index('excitation', inplace = True)
      # self.metadata_df = metadata
      
      for cubename in self.metadata.keys():
        ex = self.metadata[cubename]['ex']
        # self.metadata[cubename]['ex'] = round(float(ex), 1) if ex.isdigit() else ex
        if ex not in self.metadata_df.index:
          print(f"Attention! User has not provided metadata for cube '{ex}'")
          continue
        em_start = int(self.metadata_df.loc[ex, 'emission_start'])
        em_end = int(self.metadata_df.loc[ex, 'emission_end'])
        step = int(self.metadata_df.loc[ex, 'step'])
        self.metadata[cubename]['em_start'] = em_start
        self.metadata[cubename]['em_end'] = em_end
        self.metadata[cubename]['step'] = step
        exp = self.metadata_df.loc[ex, 'exp']
        self.metadata[cubename]['expos_val'] = float(exp) if str(exp).isdigit() else exp
        self.metadata[cubename]['notes'] = self.metadata_df.loc[ex, 'notes']
        self.metadata[cubename]['wavelengths'] = np.array(range(em_start, em_end+1, step))

  def get_metadata(self, metadata_path):
    metadata = pd.read_csv(metadata_path)
    metadata['excitation'] = metadata.excitation.astype(str)
    metadata.set_index('excitation', inplace = True)
    self.metadata_df = metadata

  def get_tls_data(self, correction_data_ls):
    # Load Correction Data (TLS Basic Wavelength Scan, several scans repetitions). All scans must have the same start, stop, step
    self.correction_LS_path = correction_data_ls
    data_files = os.listdir(correction_data_ls)
    for filename in data_files:
      measurement = filename.split('.')[0]
      data = pd.read_csv(os.path.join(correction_data_ls, filename), sep = '\t', skiprows = 9)
      wavelength = round(data.X)
      optical_power = data.Y * 10 ** 6
      data_new = pd.DataFrame({'Wavelength': wavelength, f'OP_uW_m{measurement}': optical_power})
      data_new = data_new.reset_index(drop = True).set_index('Wavelength')

      if self.correction_data_ls is None:
        self.correction_data_ls = data_new
      else:
        self.correction_data_ls = pd.concat([self.correction_data_ls, data_new], axis = 1)

    self.correction_data_ls['Average'] = self.correction_data_ls.mean(axis = 1)
    self.correction_data_all = self.correction_data_ls

    wavelengths_digit = [self.metadata[cubename]['ex'] for cubename in self.names if self.metadata[cubename]['ex'].isdigit()]
    wavelengths_needed = [float(wvl) for wvl in wavelengths_digit if float(wvl) in self.correction_data_ls.index]
    self.correction_data_ls = self.correction_data_ls.loc[wavelengths_needed]

  def get_spectral_sensitivity(self, spectral_sensitivity_data):
    self.spectral_sensitivity = pd.read_csv(spectral_sensitivity_data, index_col = 0)
  
  def process(self, cubes_to_analyse, background_cube = None, correction_data_ls = None, spectral_sensitivity_data = None):
    
    if background_cube:
      for cube in cubes_to_analyse:
        print(f"Subtracting background from '{cube}'...")
        cube_subtracted = self.raw[cube] - self.raw[background_cube]
        negatives = (cube_subtracted < 0)
        cube_subtracted[negatives] = 0
        self.processed[cube] = cube_subtracted
      print('Background subtraction done. See subtracted cubes in cubes.processed attribute.\n--------------------------------')
    else: print('Attention! No background subtraction took place.\n--------------------------------')

    if correction_data_ls:
      self.get_tls_data(correction_data_ls)

      if not self.processed:
        cubes_to_correct = self.raw.keys()
      else:
        cubes_to_correct = self.processed.keys()

      self.cubes_to_correct = cubes_to_correct
      for cubename in cubes_to_correct:
        print(f"Correcting by light source cube '{cubename}'...")
        ex = self.metadata[cubename]['ex']
        ex = float(ex) if ex.isdigit() else ex
        if ex in self.correction_data_ls.index:

          correction_factor = self.correction_data_ls['Average'][ex]/self.correction_data_ls['Average'].mean()
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

  def reshape(self, which_data = 'raw'):
    data_to_process = getattr(self, which_data)
    self.reshaped_input = which_data
    for cubename in data_to_process.keys():
      cube = data_to_process[cubename]
      cube_reshaped = np.reshape(cube, (cube.shape[0]*cube.shape[1], cube.shape[2]))
      self.reshaped[cubename] = cube_reshaped

  def get_pcs(self, cubes_to_analyse = None, components = 3, which_data = 'raw', df = False, mask_array = None, extra_transform = False, trans_factor = 0.5, trans_inplace = False):
    
    data_to_process = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = cubes_to_analyse
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
      cube_names = cubes_to_analyse
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

#=============== EEM ===============

  def get_eem(self, cubes_to_analyse = None, which_data = 'raw', mask_label = None, transform = False, plot = True, vmin = None, vmax = None, axis_ratio = None, title = None, region = None, ax = None, cbar_ax = None, fontsize = 'medium', ticksize = 'medium', xtickstep = 2, also_spectra = True):
      
    data_to_process = getattr(self, which_data)
    
    if cubes_to_analyse:
      cube_names = cubes_to_analyse
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

#============== COMBINE =====================

  def combine(self, cubes_to_analyse = None, which_data = 'raw', description = None):
    data = getattr(self, which_data)
    if cubes_to_analyse:
      cube_names = cubes_to_analyse
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
      print(f"Getting '{cubename}' from '{which_data}' data")
      cube = data[cubename]
      wavelengths = self.metadata[cubename]['wavelengths']
      if description not in self.combined:
        self.combined[description] = np.empty((cube.shape[0], cube.shape[1], 0), dtype = np.float32)
      self.combined[description] = np.concatenate((self.combined[description], cube), axis = 2)
      self.combined_metadata[description]['wavelengths'] = np.concatenate((self.combined_metadata[description]['wavelengths'], wavelengths))
    self.combined_metadata[description]['source'] = which_data
    self.combined_metadata[description]['cubes'] = cube_names

#============= SAVE TIFF =================

  def save_tiff(self, cubes_to_save = None, which_data = 'raw', mode = 'cubes', destination = None, description = None):
    
    data = getattr(self, which_data)
    
    if cubes_to_save is None:
      cubes_to_save = list(data.keys())

    if destination == None:
      destination = os.path.dirname(self.data_path)
    if description == None:
      cubenames_bases = [str.split(cubename, '.')[0] for cubename in cubes_to_save]
      description = f"Tiff_{mode.capitalize()}_{which_data.capitalize()}_{'_'.join(cubenames_bases)}"
    output_path = os.path.join(destination, description)
    os.makedirs(output_path, exist_ok=True)

    if mode == 'cubes':
      for cubename in cubes_to_save:
        cube = data[cubename]
        cube_for_tiff = cube.transpose(2, 0, 1)
        cubename_base = str.split(cubename, '.')[0]
        cube_path = os.path.join(output_path, f'{cubename_base}.tif')
        tiff.imwrite(cube_path, cube_for_tiff)
    elif mode == 'slices':
      for cubename in cubes_to_save:
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

    
    # if cubes_to_save is None:
    #   cubes_to_save = self.names
    # fromto = os.path.splitext(cubes_to_save[0])[0] + '_' + os.path.splitext(cubes_to_save[-1])[0]

    # if output_path == None:
    #   sample_path = os.path.dirname(self.data_path)
    #   if combined == True:
    #     output_path = os.path.join(sample_path, f'Tiff_Slices_{which_data.capitalize()}_Combined_{fromto}')
    #   else:
    #     output_path = os.path.join(sample_path, f'Tiff_Slices_{which_data.capitalize()}_Idividual_{fromto}')
    #   os.makedirs(output_path, exist_ok = True)
    # else: output_path = output_path

    # if combined == True:
    #   if self.combined is None:
    #     self.combine(cubes_to_save, which_data)
    #     data = self.combined
    #   else: data = self.combined
    # else:
    #   data = getattr(self, which_data)

    # if combined == True:
    #   bands_num = data.shape[2]
    #   for band in range(bands_num):
    #     band_data = data[:, :, band]
    #     filename = f'{basename}_combined_{which_data}_{fromto}_{band:04}.tif' if basename else f'Combined_{fromto}_{which_data}_{band:04}.tif'
    #     img = Image.fromarray(band_data)
    #     print(f"Saving band {band} to {os.path.join(output_path, filename)}...")
    #     img.save(os.path.join(output_path, filename), format = "TIFF")
    # else:
    #   for cubename in cubes_to_save:
    #     basename = os.path.splitext(cubename)[0]
    #     output_path_cube = os.path.join(output_path, basename)
    #     os.makedirs(output_path_cube, exist_ok = True)

    #     cube = data[cubename]
    #     bands_num = cube.shape[2]
    #     for band in range(bands_num):
    #       band_data = cube[:, :, band]
    #       wavelengths = self.metadata[cubename]['wavelengths']
    #       wvl = wavelengths[band]
    #       filename = f'{basename}_{wvl:04}.tif' if basename else f'band_{wvl:04}.tif'
    #       img = Image.fromarray(band_data)
    #       print(f"Saving band {band} to {os.path.join(output_path_cube, filename)}...")
    #       img.save(os.path.join(output_path_cube, filename), format = "TIFF")

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

  # def read_mask(self, filepath, mask_labels = None):  

  # This doesn't work yet
  # def savefile(self, name = 'cubes', path = str):
  #   filepath = os.path.join(path, name + '.pkl')
  #   with open(filepath, 'wb') as file:
  #     pickle.dump(self, file)





"""
PREVIOUS VERSION - NOW EVERYTHING ARE MAINLY IN CLASSES
"""

# import os
# import numpy as np
# import pandas as pd

# # Initiate pyimagej (at fiji mode)
# import imagej
# ij = imagej.init('sc.fiji:fiji')

# """LOAD CUBES"""
# # Version: 2024-06-25

# """
# 1. Just changed the output not to be a global variable, but ruther return a dictionary with cubes data
# 2. Also to print the cube names while loading, to see progress
# 3. Changed the data to be loaded as float, not int
# """

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

# """PROCESS CUBES"""
# # Version 2024-06-25

# """
# Introducing the alter-cube processing step
# """

# def process_cubes(cubes, cubes_to_analyse, correction_data_path = None, alter_base = None, background_cube = None):

#   global correction_data # might be useful to see the blip of that day
#   global wavelengths_needed # just for viewing in case needed

#   if background_cube:
#     for cube in cubes.keys():
#       cubes[cube]['data'] = cubes[cube]['data'] - cubes[background_cube]['data']

#   if correction_data_path:

#     # Load Correction Data (TLS Basic Wavelength Scan, several scans repetitions). All scans must have the same start, stop, step
#     file_num = [int(filename[:-4]) for filename in os.listdir(correction_data_path)]
#     anyfile_for_wavelengths = 1
#     anyfile = pd.read_csv(os.path.join(correction_data_path, str(anyfile_for_wavelengths) + '.TRQ'), sep = '\t', skiprows = 9)
#     wavelengths = round(anyfile.X, 1).astype(float)

#     correction_data = pd.DataFrame({'wavelength': wavelengths})
#     measurements = [] #seems to be not useful, maybe we can eliminate later

#     for num in file_num:
#       path = os.path.join(correction_data_path, str(num)+'.TRQ')
#       df = pd.read_csv(path, sep = '\t', skiprows = 9)
#       y = df.Y * 10 ** 6
#       colname = 'm' + str(num)
#       measurements += [colname]
#       # correction_data = pd.concat([correction_data, y], axis = 1)
#       correction_data.loc[:, colname] = y

#     correction_data.set_index(keys = 'wavelength', inplace = True)

#     # Select only those wavelengths which are corresponding tou our cubes of interest
#     wavelengths_needed = [cubes[cube]['ex'] for cube in cubes_to_analyse if cubes[cube]['ex'] in list(wavelengths)]
#     correction_data = correction_data.loc[wavelengths_needed]

#     correction_data['average'] = correction_data[measurements].mean(axis = 1)

#     for cube in cubes.keys():
#       ex = cubes[cube]['ex']
#       if ex in list(wavelengths_needed): #maybe this is odd, i don't remember, will check later

#         correction_factor = correction_data['average'][ex]/correction_data['average'].mean()
#         cubes[cube]['correction_factor'] = round(correction_factor, 2)

#         data_corrected = cubes[cube]['data'] / correction_factor
#         cubes[cube]['data_corrected'] = np.around(data_corrected, decimals = 2)

#       else: print(f"Cube [{cube}] does not have a correction factor, but sometimes that's ok! ;)")

#   if alter_base:
#     for cube in cubes_to_analyse:
#       altercube = cubes[cube]['data_corrected'] / (cubes[alter_base]['data_corrected'] + 1)
#       cubes[cube]['alter_base'] = alter_base
#       cubes[cube]['altercube'] = np.around(altercube, decimals = 4)


# """GET ROIs"""
# # Version 2024-06-26 (2)

# """
# Turned the function into a class, for importing and reusing convenience
# Some uppercases to lowercases
# """

# class Regions():

#   def __init__(self):
#     self.regions = {}

#   def add_roi(self, region, cubes, cubes_to_analyse, y1, y2, x1, x2):
    
#     self.regions[region] = {}
    
#     rows = slice(y1, y2+1)
#     cols = slice(x1, x2+1)

#     self.regions[region]['rows'] = rows
#     self.regions[region]['cols'] = cols
#     self.regions[region]['rows_str'] = f'{rows.start}:{rows.stop}'
#     self.regions[region]['cols_str'] = f'{cols.start}:{cols.stop}'
#     self.regions[region]['num_pixels'] = (rows.stop - rows.start) * (cols.stop - cols.start)

#     spectra_corrected = {}
#     spectra_corrected['bycube'] = {}
#     spectra_corrected['combined'] = {}

#     eems_corrected = {}

#     ###
#     spectra = {}
#     spectrum = {}

#     for cube in cubes_to_analyse:

#       # Get wavelengths from cubes metadata
#       # This can be transferred to the cubes metadata, and later just accessed like cubes[cube]['wavelengths']
#       start = cubes[cube]['em_start']
#       end = cubes[cube]['em_end']
#       step = cubes[cube]['step']
#       wavelengths = range(start, end+1, step)

#       # Slice each cube to the region's coordinates
#       cube_segment = cubes[cube]['data_corrected'][rows, cols, :]

#       # Reshape each cube and get 2D data for spectra for each pixel
#       cube_reshaped = cube_segment.reshape(cube_segment.shape[0] * cube_segment.shape[1], cube_segment.shape[2])
#       spectra[cube] = pd.DataFrame(cube_reshaped, columns = wavelengths)

#       # Get the average spectra by pixels for each cube
#       cube_averaged_byPixel = np.mean(cube_segment, axis = (0, 1), keepdims = True).reshape(1, cube_segment.shape[2])
#       spectrum[cube] = pd.DataFrame(cube_averaged_byPixel, columns = wavelengths, index = [int(cubes[cube]['ex'])])

#       # Get EEM out of averaged (over pixels) spectra, by stacking them together
#       try:
#         eem_averaged_corrected
#       except NameError:
#         eem_averaged_corrected = pd.DataFrame(columns = wavelengths)

#       eem_averaged_corrected = pd.concat([eem_averaged_corrected, spectrum[cube]], axis = 0).sort_index(ascending = False)

#     spectra_corrected['bycube']['all'] = spectra
#     spectra_corrected['bycube']['average'] = spectrum

#     self.regions[region]['eem_averaged_corrected'] = eem_averaged_corrected

#     spectra_corrected['combined']['all'] = None
#     spectra_corrected['combined']['average'] = None

#     self.regions[region]['spectra_corrected'] = spectra_corrected 


# """TEST DATA"""
# # Version 2024-06-24

# project = '/content/drive/My Drive/DATA'
# experiment = 'Test Data' # the same with less cubes
# sample = 'right_leg_tendon_nerve'
# dataset = 'cubes'

# test_data = {
#     'project' : '/content/drive/My Drive/DATA',
#     'experiment' : 'Test Data',
#     'sample' : 'right_leg_tendon_nerve',
#     'dataset' : 'cubes',
#     'data_path' : os.path.join(project, experiment, sample, dataset),
#     'metadata_path' : os.path.join(project, experiment, sample, 'metadata.csv'),
#     'correction_data_path' : os.path.join(project, experiment, sample, 'correction_data'),
# }

# project = '/content/drive/My Drive/DATA'
# experiment = 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve'
# sample = 'right_leg_tendon_nerve'
# dataset = 'cubes'

# real_data = {
#     'project' : '/content/drive/My Drive/DATA',
#     'experiment' : 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve',
#     'sample' : 'right_leg_tendon_nerve',
#     'dataset' : 'cubes',
#     'data_path' : os.path.join(project, experiment, sample, dataset),
#     'metadata_path' : os.path.join(project, experiment, sample, 'metadata.csv'),
#     'correction_data_path' : os.path.join(project, experiment, sample, 'correction_data.lnk'),
# }

# some_data = {'test': test_data, 'real': real_data}

