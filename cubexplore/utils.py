import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

# deleted from cubes.py | imported
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

# deleted from cubes.py | imported
def ensure_list(input_value):
  if isinstance(input_value, list): # check if list
    return input_value
  elif isinstance(input_value, type(dict().keys())): # check if dict keys
    return list(input_value)
  elif isinstance(input_value, str): # check if string
    return [input_value]
  else:
    raise TypeError("Input must be either a string or a list")

def bin_mask(mask, bin_size):
  rows, cols = mask.shape
  if rows % bin_size != 0 or cols % bin_size != 0:
    rows = rows - rows % bin_size
    cols = cols - cols % bin_size
    mask = mask[:rows, :cols]
  
  zoom_factor = 1 / bin_size
  return zoom(mask, zoom_factor, order=0)

# Read a Google Sheet
# deleted from 
def read_sheet(url, sheet, skipr=0, dropna=False, dropna_axis='rows'):
  doc_id = url.split('/d/')[-1].split('/')[0]
  url = f'https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&sheet={sheet}'
  if dropna==True:
    df = pd.read_csv(url, skiprows = skipr).dropna(axis = dropna_axis, how = 'all')
  else:
    df = pd.read_csv(url, skiprows = skipr)
  return df

