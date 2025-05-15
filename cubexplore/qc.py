import os
import pandas as pd

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