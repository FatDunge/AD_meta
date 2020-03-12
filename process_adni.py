#%% 
#Rename
import os
import re
import pandas as pd
import csv

df = pd.read_csv('./data/center_info/ADNI/ADNIMERGE.csv')
#%%
f = open(r'G:\ADNI_name_changing.txt', mode='a+')
site = 'screening'
pattern = '\d{3}_S_\d{4}'
path = r'G:\proed'
filenames = os.listdir(path)
for filename in filenames:
    if 'mwp1' in filename or '.xml' in filename:
        number = re.search(pattern, filename, flags=0).group()
        for index, row in df.iterrows():
            if row['PTID'] == number and row['VISCODE'] == 'bl':
                l = row['DX_bl']
                if isinstance(l, str):
                    if l == 'AD':
                        label = 'AD'
                    elif l == 'CN':
                        label = 'NC'
                    elif 'MCI' in l:
                        label = 'MC'
                else:
                    print('{} label error'.format(row['PTID']))
                src = os.path.join(path, filename)
                if 'mwp1' in filename:
                    suffix = 'nii'
                elif '.xml' in filename:
                    suffix = 'xml'
                dist_name = 'mwp1_{}_{}_{}_{}.{}'.format(row['PTID'], label,
                                                         row['ORIGPROT'], site, suffix)
                dist = os.path.join(path, dist_name)
                print('{}->{}'.format(filename, dist_name), file=f)
                
                csv_path = os.path.join(path, '{}_{}_{}_{}.csv'.format(row['PTID'], label,
                                                                    row['ORIGPROT'], site))
                with open(csv_path, 'w', newline='') as file:
                    fieldnames = ['age', 'male', 'female', 'MMSE']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    if row['PTGENDER'] == 'Male':
                        male = 1
                    else: 
                        male = 0
                    female = 1 - male

                    writer.writerow({'age': row['AGE']/100,
                                    'male': male, 'female':female,
                                    'MMSE':row['MMSE']})
                try:
                    os.rename(src, dist)
                except FileExistsError:
                    print(row['PTID'])
    elif '.mat' in filename:
        os.remove(os.path.join(path, filename))

f.close()
#%%
import shutil
path = './data/report'
filenames = os.listdir(path)
to = './data/AD/ADNI'
pattern = '\d{3}_S_\d{4}'
for filename in filenames:
    src = os.path.join(path, filename)
    center = re.search(pattern, filename, flags=0).group()[:3]
    if '.csv' in filename:
        dst = os.path.join(to, center, 'personal_info', filename)
    elif '.xml' in filename:
        dst = os.path.join(to, center, 'report', filename)
    shutil.move(src,dst)

#%%
#gen filename
import os

data_path = './data/AD/ADNI'

center_names = os.listdir(data_path)
for center_name in center_names:
    path = os.path.join(data_path, center_name, 'personal_info')
    filenames = os.listdir(path)
    for filename in filenames:
        with open(os.path.join(data_path, center_name, 'filenames.txt'), 'a') as txt:
            print(filename[:-4], file=txt)
#%%
import os
import shutil
data_path = r'G:\tmp'

names = os.listdir(data_path)
for name in names:
    if os.path.isdir(os.path.join(data_path, name)):
        path = os.path.join(data_path, name)
        mri_path = os.path.join(path, 'mri')
        pif_path = os.path.join(path, 'personal_info')
        rp_path = os.path.join(path, 'report')
        os.mkdir(mri_path)
        os.mkdir(os.path.join(path, 'mri_smoothed'))
        os.mkdir(os.path.join(path, 'csv'))
        os.mkdir(os.path.join(path, 'csv_removed'))
        os.mkdir(os.path.join(path, 'mri_smoothed_removed'))
        os.mkdir(pif_path)
        os.mkdir(rp_path)

        filenames = os.listdir(data_path)
        for filename in filenames:
            if not os.path.isdir(name):
                if filename[0:3] == name or filename[5:8] == name:
                    if '.nii' in filename:
                        src = os.path.join(data_path, filename)
                        dst = os.path.join(mri_path, filename)
                        shutil.move(src,dst) 
                    elif '.csv' in filename:
                        src = os.path.join(data_path, filename)
                        dst = os.path.join(pif_path, filename)
                        shutil.move(src,dst) 
                    elif '.xml' in filename:
                        src = os.path.join(data_path, filename)
                        dst = os.path.join(rp_path, filename)
                        shutil.move(src,dst) 

#%%
import datasets

centers = datasets.load_centers_adni(use_nii=True)

# %%
import mask
_mask = mask.Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')
for center in centers:
    for person in center.persons:
        person.create_csv(_mask)

# %%
path = r'G:\tmp'
filenames = os.listdir(path)
for filename in filenames:
    if '.csv' in filename:
        center = filename[0:3]
        try:
            os.mkdir(os.path.join(path, center))
        except:
            pass
#%%
import datasets
from mask import Mask

_mask = Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')
centers = datasets.load_centers_adni(use_nii=True, nii_prefix='mri_smoothed/{}.nii')

# %%
for center in centers:
    for person in center.persons:
        person.create_csv(_mask)

# %%
