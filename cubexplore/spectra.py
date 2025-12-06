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

