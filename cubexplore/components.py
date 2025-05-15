import os
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image

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
      # img = Image.open(component_path)
      img = tiff.imread(component_path)
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

def get_mask_from_components(unmixing_path, n_components = 3, source = 'component_data', component_values: dict = {1:200, 2:100, 3:0}, composite = True, normalize = False, threshold_quantiles = None):
  
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

# deleted from utils.py
# deleted from cubes.py
def read_spectral_library(library_path):
  components = [f'C{i}' for i in range(1, 11)]
  spectral_library = pd.read_csv(library_path, sep = '\t', skiprows = 1)
  spectral_library = spectral_library.iloc[:, 1:12]
  spectral_library = spectral_library.T.reset_index().T
  spectral_library.columns = ['wavelength'] + components
  spectral_library = spectral_library.reset_index(drop = True)
  spectral_library = spectral_library.astype(float)
  return spectral_library