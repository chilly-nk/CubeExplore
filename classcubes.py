import os
import numpy as np
import pandas as pd
import datetime

# Initiate pyimagej (at fiji mode)
import imagej
ij = imagej.init('sc.fiji:fiji')

class Cubes:
  
  def __init__(self, data_path, metadata_path = None, cubes_to_analyse = None):
    
    self.last_loaded = datetime.datetime.now()
    
    self.raw = {}
    self.metadata = {}
    
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
          print(f'Attention! Condition "{ex}" not in metadata.')
          continue
        self.metadata[cube]['em_start'] = int(metadata.loc[ex, 'emission_start'])
        self.metadata[cube]['em_end'] = int(metadata.loc[ex, 'emission_end'])
        self.metadata[cube]['step'] = int(metadata.loc[ex, 'step'])
        exp = metadata.loc[ex, 'exp']
        self.metadata[cube]['expos_val'] = float(exp) if str(exp).isdigit() else exp
        self.metadata[cube]['notes'] = metadata.loc[ex, 'notes']