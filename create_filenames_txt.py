# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import os

data_path = './data/AD/EDSD/EDSD_T1'

center_names = os.listdir(data_path)
for center_name in center_names:
    report_path = os.path.join(data_path, center_name, 'report')
    filenames = os.listdir(report_path)
    for filename in filenames:
        if '.xml' in filename:
            with open(os.path.join(data_path, center_name, 'filenames.txt'), 'a') as txt:
                print(filename[4:-4], file=txt)

#%%
data_path = './data/AD/MCAD/AD_S03/AD_S03_MPR'


report_path = os.path.join(data_path, 'report')
filenames = os.listdir(report_path)
for filename in filenames:
    if '.xml' in filename:
        with open(os.path.join(data_path,  'filenames.txt'), 'a') as txt:
            print(filename[4:-4], file=txt)

#%%
