#%%
import datasets
from mask import Mask
import numpy as np

center_list = datasets.load_centers_mcad(use_nii=True,
                                         nii_prefix='mri_smoothed/{}.nii')
mask = Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')

# %%
"""
for center in center_list:
    for person in center.persons:
        person.create_csv(mask)
"""
# %%
import datasets
from mask import Mask
import numpy as np
from scipy.stats import norm
from meta import gen_roi_studies
from meta import meta
from meta import show_forest
import csv
import pandas as pd

roi_name_path = './data/mask/BNA_subregions.csv'
df = pd.read_csv(roi_name_path, index_col=0)

settings = {"datatype":"CONT",
            "models":"Random",
            "algorithm":"IV-Cnd",
            "effect":"SMD"}

centers = ['ALL']
label = ['NC', 'MCI', 'AD']
groups = [(2,0), (2,1), (1,0)]
csv_prefix = 'csv/{}.csv'

center_edsd = datasets.load_centers_edsd(use_csv=True,
                                         csv_prefix=csv_prefix)
center_mcad = datasets.load_centers_mcad(use_csv=True,
                                         csv_prefix=csv_prefix)
center_adni = datasets.load_centers_adni(use_csv=True,
                                         csv_prefix=csv_prefix)
rois = [str(i) for i in range(1,247)]

for center in centers:
    if center == 'MCAD':
        center_list = center_mcad
    elif center == 'EDSD':
        center_list = center_edsd
    else:
        center_list = center_edsd + center_mcad + center_adni
    for group in groups:
        eg = group[0]
        cg = group[1]
        studies = gen_roi_studies(center_list, rois, eg, cg)
        results = {}
        csv_path = './report/roi/brainnetome/{}_{}{}.csv'.format(center, label[eg], label[cg])
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['ID', 'NAME' ,'ES', 'LCI', 'UCI', 'SE', 'I^2', 'Z', 'P', 'Sign']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for roi in rois:
                name = df.loc[int(roi)]['name']
                results[roi] = meta(studies[roi], settings)
                es = results[roi][0][1]
                lci = results[roi][0][3]
                uci = results[roi][0][4]
                se = results[roi][0][6]
                i2 = results[roi][0][9]
                z = results[roi][0][10]
                p = norm.sf(z) * 2
                if p < 0.001/246:
                    sign = '***'
                elif p < 0.01/246:
                    sign = '**'
                elif p < 0.05/246:
                    sign = '*'
                else:
                    sign = 'NS'

                
                
                writer.writerow({'ID': roi, 'NAME': name,'ES': es,
                                'LCI': lci, 'UCI': uci,
                                'SE': se, 'I^2': i2,
                                'Z': z, 'P': p, 'Sign': sign})

                if (int(roi) in [1, 100, 153, 215]) and eg == 2 and cg==0:
                    show_forest(results[roi])
#%%
#------------------------------------------------------------
# Confound remove
import datasets
from mask import Mask
import numpy as np
x = []
y = []
center_list = datasets.load_centers_mcad(use_nii=False, use_csv=True,
                                         use_personal_info=True,
                                         use_xml=True)
for center in center_list:
    for person in center.persons:
        personal_info = person.get_presonal_info_values()[0:3]
        tiv = person.get_tiv()
        intercept = 1
        label = person.get_label_binary()
        x.append(np.hstack((personal_info, tiv, intercept, label)))
        y.append(person.dataframe.values.flatten())
x = np.array(x)
y = np.array(y)

x_inv = np.linalg.pinv(x)
beta = np.dot(x_inv, y)
np.save('./npy/csv_beta_mcad.npy', beta)

beta_a = beta[:4]

for center in center_list:
    for person in center.persons:
        personal_info = person.get_presonal_info_values()[0:3]
        tiv = person.get_tiv()
        x = np.hstack((personal_info, tiv))
        y = person.dataframe.values.flatten()
        y_hat = np.dot(x, beta_a)
        y_new = y - y_hat
        person.create_other_csv(y_new)
#%%
from mask import Mask
import pandas as pd
csv_path = './report/roi/brainnetome/ALL_ADMCI.csv'

mask = Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')
df = pd.read_csv(csv_path, index_col=0)

# %%
df

# %%
