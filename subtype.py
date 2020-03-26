#%%
import datasets
centers_mcad = datasets.load_centers_mcad(use_nii=False, use_csv=True,
                                          use_personal_info=True, use_xml=True)
centers_edsd = datasets.load_centers_edsd(use_nii=False, use_csv=True,
                                          use_personal_info=True, use_xml=True)
centers_adni = datasets.load_centers_adni(use_nii=False, use_csv=True,
                                          use_personal_info=True, use_xml=True)

center_list = centers_mcad + centers_edsd + centers_adni
#%%
for center in center_list:
    center.save_labels('origin.csv')
# %%
#prepare HYDRA csv
import csv
feature_path = './matlab/HYDRA/data/features.csv'
covariate_path = './matlab/HYDRA/data/covariate.csv'
i = 0
with open(feature_path, 'w', newline='') as feature_file:
    features = center_list[0].persons[0].dataframe.values.flatten().tolist()
    header = ['id'] + ['feature_{}'.format(i) for i in range(len(features))] + ['group']
    featureswriter = csv.DictWriter(feature_file, fieldnames=header)
    featureswriter.writeheader()
    with open(covariate_path, 'w', newline='') as covariate_file:
        covariatewriter = csv.writer(covariate_file)
        for center in center_list:
            for person in center.persons:
                if person.label == 2:
                    label = 1
                elif person.label == 0:
                    label = -1
                else:
                    continue
                features = person.dataframe.values.flatten().tolist()
                feature_row = [i] + features + [label]
                dictionary = dict(zip(header, feature_row))

                covariate_row = [i] + person.get_presonal_info_values().tolist()[0:3] + [person.get_tiv()]
                featureswriter.writerow(dictionary)
                covariatewriter.writerow(covariate_row)
                i += 1


# %%
from scipy import io
import numpy as np

mat = io.loadmat('./matlab/HYDRA/result/HYDRA_results.mat')

#%%
mat
# %%
CIDX = mat['CIDX']
#%%
CIDX = np.transpose(CIDX)
#%%
CIDX = CIDX[1]
#%%
print(np.unique(CIDX))
#%%
CIDX.shape
#%%
CIDX = CIDX + 1

# %%
i = 0
for center in center_list:
    for person in center.persons:
        if person.label == 2 or person.label == 0:
            person.label = CIDX[i]
            i += 1

# %%
for center in center_list:
    for person in center.persons:
        print('{}:{}'.format(person.filename, person.label))

# %%
for center in center_list:
    center.save_labels('HYDRA.csv')

# %%
