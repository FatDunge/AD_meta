#%%
import datasets
center_list = datasets.load_centers_all()
# %%
#prepare HYDRA csv
import csv
import numpy as np
feature_path = './matlab/HYDRA/data/features.csv'
covariate_path = './matlab/HYDRA/data/covariate.csv'
i = 0
with open(feature_path, 'w', newline='') as feature_file:
    header = ['id'] + ['feature_{}'.format(i) for i in range(456)] + ['group']
    featureswriter = csv.DictWriter(feature_file, fieldnames=header)
    featureswriter.writeheader()
    with open(covariate_path, 'w', newline='') as covariate_file:
        covariatewriter = csv.writer(covariate_file)
        for center in center_list:
            features_rv, labels = center.get_csv_values(flatten=True)
            features_ct, _ = center.get_csv_values(prefix='cortical_thickness/{}.csv',
                                      flatten=True)
            features = np.concatenate([features_rv, features_ct], axis=1)
            personal_infos, _ = center.get_presonal_info_values()
            tcgws, _  = center.get_tivs_cgws()
            for feature, label, personal_info, tcgw  in zip(features, labels, personal_infos, tcgws):
                if label == 2:
                    label = 1
                elif label == 0:
                    label = -1
                else:
                    continue
                feature_row = [i] + feature.tolist() + [label]
                dictionary = dict(zip(header, feature_row))

                covariate_row = [i] + personal_info[0:3].tolist() + [tcgw.tolist()[0]]
                featureswriter.writerow(dictionary)
                covariatewriter.writerow(covariate_row)
                i += 1


# %%
from scipy import io
import numpy as np

mat = io.loadmat('./matlab/HYDRA/result/HYDRA_results.mat')

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
    center.save_labels('HYDRA.csv')

# %%
