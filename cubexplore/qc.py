import os
import pandas as pd

#==== CHECK CHARACTERS & CORRECT PATHS ===================
#==== 2025-12-06

class CharCheck:
  def __init__(self, directory, character=" "):
    self.directory = directory
    self.character = character
    # self.files(character)
    # self.dirs(character) # turning these off, because the self.last_checked takes only dirs, which can make a confusion. so its better to do explicitly

  def items(self, which_items='files', character=" ", max_length=30, topdown=False):
    self.character = character
    self.items_to_rename = []
    self.paths_to_rename = []
    for root, dirs, files in os.walk(self.directory, topdown=topdown):
      targets = []
      if which_items == 'files':
        targets = files
      elif which_items == 'dirs':
        targets = dirs
      for target in targets:
        if character in target:
          self.items_to_rename.append(target)
          self.paths_to_rename.append(os.path.join(root, target))
    self.last_checked = which_items
    return self

  def files(self, character=" ", max_length=30, topdown=False):
    self.items(which_items='files', character=character, max_length=max_length, topdown=topdown)
    return self

  def dirs(self, character=" ", max_length=30, topdown=False):
    self.items(which_items='dirs', character=character, max_length=max_length, topdown=topdown)
    return self

  def rename(self, new_character="_"):
    for path in self.paths_to_rename:
      name = os.path.basename(path)
      new_name = name.replace(self.character, new_character)
      new_path = os.path.join(os.path.dirname(path), new_name)
      if os.path.exists(new_path):
        print(f"Warning: '{new_name}' already exists!")
        choice = input(f"Do you really want to overwrite '{new_name}' with '{name}'? (Y/N): ").strip().lower()

        if choice != 'y':
          print(f"Rename operation canceled for '{name}'.")
          continue
        else:
          os.rename(path, new_path)
          print(f"Renamed {self.last_checked[:-1]}: {name} -> {new_name}")
      else:
        os.rename(path, new_path)
        print(f"Renamed {self.last_checked[:-1]}: {name} -> {new_name}")

    return self

#==== SAMPLE NAME CHECK IN DATASET ============

class SampleNames:
  def __init__(self, experiment_path, ref_item='Cube', samplename_positions={'Cube': 1, 'Scale': 1, 'Metadata': 1, 'Mask': 2, 'TTC': 1}):
    self.experiment_path = experiment_path
    self.ref_item = ref_item
    self.positions = samplename_positions
    
    self.ref_items = self.get_items(item_type=ref_item)
    self.ref_samples = self.get_samples(item_type=ref_item)
    # self.checked_items
    # self.checked_samples
    # self.checked_status

  def get_items(self, item_type):
    return sorted([name for name in os.listdir(self.experiment_path) if item_type in name])
  
  def get_samples(self, item_type):
    items = self.get_items(item_type)
    return sorted(list(set(['_'.join(instance.split('.')[0].split('_')[self.positions[item_type]:]) for instance in items if instance.split('.')[0].split('_')[self.positions[item_type]:]])))

  def check(self, item_type):
    items = self.get_items(item_type)
    samples = self.get_samples(item_type)
    self.checked_samples = samples
    
    if items:
      self.checked_status = [s in samples for s in self.ref_samples]
    else:
      self.checked_status = None
    return self

  def qc(self, verbose=True, to_csv=False):
    self.status = pd.DataFrame({'Samples': self.ref_samples})
    if verbose:
      if (self.qc(verbose=False).status==False).values.any():
        print('SampleName consistency might be NOT OK! :(((')
      else:
        print('SampleName consistency is OK ;)')
      print('----------------------')
    
    for item_type in self.positions.keys():
      if verbose:
        print(f'{item_type}: {self.check(item_type).checked_samples}')
      self.status[item_type] = self.check(item_type).checked_status
    
    if verbose:
      print('----------------------')

    if to_csv:
      self.status.to_csv(os.path.join(self.experiment_path, 'QC_SampleNames.csv'), index=False)
    
    return self

  def __repr__(self):
    return f'SampleNames:\n Dataset={self.experiment_path.split("/")[-1]},\n ref_item={self.ref_item},\n samplename_positions={self.positions}'