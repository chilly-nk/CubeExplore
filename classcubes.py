import os
import numpy as np
import pandas as pd
import datetime

# Initiate pyimagej (at fiji mode)
import imagej
ij = imagej.init('sc.fiji:fiji')

class Cubes:
  
  def __init__(self, data_path, metadata_path = None):
    self.last_loaded = datetime.datetime.now()
    self.data_path = data_path
    self.metadata_path = metadata_path
    
    self.raw = {}
    self.metadata = {}
    
    cube_names = sorted(os.listdir(data_path))
    for i in cube_names:
      # cubes[i] = tiff.imread(os.path.join(data_path, i)).transpose(1, 2, 0) # for loading cubes from 3D tiffs
      print(f'Loading {i}...')
      img = ij.io().open(os.path.join(data_path, i))
      # self.raw[cube_name] = 
      cube = np.array(ij.py.from_java(img), dtype = np.float32)
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

      self.raw[i] = cube_data