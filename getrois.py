"""GET ROIs"""
# Version 2024-06-26 (2)

"""
Turned the function into a class, for importing and reusing convenience
Some uppercases to lowercases
"""

class Regions():

  def __init__(self):
    self.regions = {}

  def add_roi(self, region, cubes, cubes_to_analyse, y1, y2, x1, x2):
    
    self.regions[region] = {}
    
    rows = slice(y1, y2+1)
    cols = slice(x1, x2+1)

    self.regions[region]['rows'] = rows
    self.regions[region]['cols'] = cols
    self.regions[region]['rows_str'] = f'{rows.start}:{rows.stop}'
    self.regions[region]['cols_str'] = f'{cols.start}:{cols.stop}'
    self.regions[region]['num_pixels'] = (rows.stop - rows.start) * (cols.stop - cols.start)

    spectra_corrected = {}
    spectra_corrected['bycube'] = {}
    spectra_corrected['combined'] = {}

    eems_corrected = {}

    ###
    spectra = {}
    spectrum = {}

    for cube in cubes_to_analyse:

      # Get wavelengths from cubes metadata
      # This can be transferred to the cubes metadata, and later just accessed like cubes[cube]['wavelengths']
      start = cubes[cube]['em_start']
      end = cubes[cube]['em_end']
      step = cubes[cube]['step']
      wavelengths = range(start, end+1, step)

      # Slice each cube to the region's coordinates
      cube_segment = cubes[cube]['data_corrected'][rows, cols, :]

      # Reshape each cube and get 2D data for spectra for each pixel
      cube_reshaped = cube_segment.reshape(cube_segment.shape[0] * cube_segment.shape[1], cube_segment.shape[2])
      spectra[cube] = pd.DataFrame(cube_reshaped, columns = wavelengths)

      # Get the average spectra by pixels for each cube
      cube_averaged_byPixel = np.mean(cube_segment, axis = (0, 1), keepdims = True).reshape(1, cube_segment.shape[2])
      spectrum[cube] = pd.DataFrame(cube_averaged_byPixel, columns = wavelengths, index = [int(cubes[cube]['ex'])])

      # Get EEM out of averaged (over pixels) spectra, by stacking them together
      try:
        eem_averaged_corrected
      except NameError:
        eem_averaged_corrected = pd.DataFrame(columns = wavelengths)

      eem_averaged_corrected = pd.concat([eem_averaged_corrected, spectrum[cube]], axis = 0).sort_index(ascending = False)

    spectra_corrected['bycube']['all'] = spectra
    spectra_corrected['bycube']['average'] = spectrum

    self.regions[region]['eem_averaged_corrected'] = eem_averaged_corrected

    spectra_corrected['combined']['all'] = None
    spectra_corrected['combined']['average'] = None

    self.regions[region]['spectra_corrected'] = spectra_corrected 