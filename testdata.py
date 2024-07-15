import os

"""TEST DATA"""
# Version 2024-06-24

project = '/content/drive/My Drive/DATA'
experiment = 'Test Data' # the same with less cubes
sample = 'right_leg_tendon_nerve'
dataset = 'cubes'

test_data = {
    'project' : '/content/drive/My Drive/DATA',
    'experiment' : 'Test Data',
    'sample' : 'right_leg_tendon_nerve',
    'dataset' : 'cubes',
    'data_path' : os.path.join(project, experiment, sample, dataset),
    'metadata_path' : os.path.join(project, experiment, sample, 'metadata.csv'),
    'correction_data_path' : os.path.join(project, experiment, sample, 'correction_data'),
}

project = '/content/drive/My Drive/DATA'
experiment = 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve'
sample = 'right_leg_tendon_nerve'
dataset = 'cubes'

real_data = {
    'project' : '/content/drive/My Drive/DATA',
    'experiment' : 'Exp 2024-05-03 - 4D HSI Rat Tendon + Nerve',
    'sample' : 'right_leg_tendon_nerve',
    'dataset' : 'cubes',
    'data_path' : os.path.join(project, experiment, sample, dataset),
    'metadata_path' : os.path.join(project, experiment, sample, 'metadata.csv'),
    'correction_data_path' : os.path.join(project, experiment, sample, 'correction_data.lnk'),
}

some_data = {'test': test_data, 'real': real_data}