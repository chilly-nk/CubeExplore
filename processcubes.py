"""PROCESS CUBES"""
# Version 2024-06-25

"""
Introducing the alter-cube processing step
"""

def process_cubes(cubes, cubes_to_analyse, correction_data_path = None, alter_base = None, background_cube = None):

  global correction_data # might be useful to see the blip of that day
  global wavelengths_needed # just for viewing in case needed

  if background_cube:
    for cube in cubes.keys():
      cubes[cube]['data'] = cubes[cube]['data'] - cubes[background_cube]['data']

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

    # Select only those wavelengths which are corresponding tou our cubes of interest
    wavelengths_needed = [cubes[cube]['ex'] for cube in cubes_to_analyse if cubes[cube]['ex'] in list(wavelengths)]
    correction_data = correction_data.loc[wavelengths_needed]

    correction_data['average'] = correction_data[measurements].mean(axis = 1)

    for cube in cubes.keys():
      ex = cubes[cube]['ex']
      if ex in list(wavelengths_needed): #maybe this is odd, i don't remember, will check later

        correction_factor = correction_data['average'][ex]/correction_data['average'].mean()
        cubes[cube]['correction_factor'] = round(correction_factor, 2)

        data_corrected = cubes[cube]['data'] / correction_factor
        cubes[cube]['data_corrected'] = np.around(data_corrected, decimals = 2)

      else: print(f"Cube [{cube}] does not have a correction factor, but sometimes that's ok! ;)")

  if alter_base:
    for cube in cubes_to_analyse:
      altercube = cubes[cube]['data_corrected'] / (cubes[alter_base]['data_corrected'] + 1)
      cubes[cube]['alter_base'] = alter_base
      cubes[cube]['altercube'] = np.around(altercube, decimals = 4)