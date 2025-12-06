import numpy as np
import pandas as pd
from pathlib import Path

def read_spectra(datapath, wvl, sample, skiprows=13):
  datapath = Path(datapath)
  files = datapath.glob(f'{wvl}_{sample}*.txt')
  dfs = []
  variant = f'{wvl}_{sample}'
  for f in files:
    wvl = f.name.split('_')[0]
    df = pd.read_csv(f, sep='\t', skiprows=skiprows)
    df.columns = ['Wavelength', f'Replica']
    df.set_index('Wavelength', inplace=True)
    dfs.append(df)
  df = pd.concat(dfs, axis=1)
  df[variant] = df.mean(axis=1)
  df = df[[variant]]

  return df

def read_maya(path, skiprows=13):
  path = Path(path)
  filepaths = sorted(path.glob("*.txt"))

  dfs = []
  for fp in filepaths:
    df = pd.read_csv(fp, skiprows=skiprows, sep='\t', header=None).set_index(0)
    dfs.append(df)
  data = pd.concat(dfs, axis=1)

  data.reset_index(inplace=True)
  data.columns = ['wavelength'] + [f"replicate_{i+1}" for i in range(len(dfs))]
  # data_long = data.melt(id_vars='wavelength', value_vars=data.columns, var_name='replicate', value_name='intensity')
  data.set_index('wavelength', inplace=True)
  data['average_intensity'] = data.mean(axis=1)
  return data

class FWHM():
  def __init__(self, x: np.ndarray, y: np.ndarray):
    """
    Parameters:
    x: 1-D array of wavelengths
    y: 1-D array of intensities. Must correspond to a spectrum with exactly one peak.
    """

    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    half_max = peak_y / 2

    left_idx = np.where(y[:peak_idx] < half_max)[0]
    if len(left_idx) > 0:
      i1 = left_idx[-1]
      i2 = i1 + 1
    x_left = np.interp(half_max, [y[i1], y[i2]], [x[i1], x[i2]])

    right_idx = np.where(y[peak_idx:] < half_max)[0]
    if len(right_idx) > 0:
      i2 = peak_idx + right_idx[0]
      i1 = i2 - 1
    x_right = np.interp(half_max, [y[i1], y[i2]], [x[i1], x[i2]])

    fwhm = x_right - x_left

    fwhm_center = x_left + fwhm / 2
    peak_offset = fwhm_center - peak_x

    self.x = x
    self.y = y
    self.peak_idx = peak_idx
    self.peak_x = peak_x
    self.peak_y = peak_y
    self.half_max = half_max
    self.left_idx = left_idx
    self.x_left = x_left
    self.right_idx = right_idx
    self.x_right = x_right
    self.fwhm = fwhm
    self.fwhm_center = fwhm_center
    self.peak_offset = peak_offset

