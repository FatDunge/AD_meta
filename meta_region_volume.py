#%%
import datasets
import numpy as np
import os
import csv
from meta_analysis import utils

centers = datasets.load_centers_all()
out_path = './results/meta'
threshold = 10
label_pairs = [(2, 0)]
csv_prefixs = ['csv/', 'cortical_thickness/']
for label_pair in label_pairs:
    for csv_prefix in csv_prefixs:
        label_eg, label_cg = label_pair
        mean_egs = []
        std_egs = []
        count_egs = []
        mean_cgs = []
        std_cgs = []
        count_cgs = []
        
        center_names = []
        for center in centers:
            persons_eg = center.get_by_label(label_eg)
            persons_cg = center.get_by_label(label_cg)
            
            if len(persons_eg) > threshold and len(persons_cg) > threshold:
                features_eg, _ = center.get_csv_values(persons=persons_eg,
                                                        prefix=csv_prefix+'{}.csv',
                                                        flatten=True)
                features_cg, _ = center.get_csv_values(persons=persons_cg,
                                                        prefix=csv_prefix+'{}.csv',
                                                        flatten=True)
                mean_eg, std_eg, n_eg = utils.cal_mean_std_n(features_eg)
                mean_cg, std_cg, n_cg = utils.cal_mean_std_n(features_cg)
                mean_egs.append(mean_eg)
                std_egs.append(std_eg)
                count_egs.append(n_eg)
                mean_cgs.append(mean_cg)
                std_cgs.append(std_cg)
                count_cgs.append(n_cg)
                center_names.append(center.name)

        
        mean_egs = np.stack(mean_egs)
        std_egs = np.stack(std_egs)
        count_egs = np.stack(count_egs)
        mean_cgs = np.stack(mean_cgs)
        std_cgs = np.stack(std_cgs)
        count_cgs = np.stack(count_cgs)

        mean_egs_T = mean_egs.T
        std_egs_T = std_egs.T
        mean_cgs_T = mean_cgs.T
        std_cgs_T = std_cgs.T

        i = 0
        for ems, ess, cms, css in zip(mean_egs_T, std_egs_T,
                                        mean_cgs_T, std_cgs_T):
            i += 1
            csv_path = os.path.join(out_path, csv_prefix, 'feature{}.csv'.format(i))
            with open(csv_path, 'w', newline='') as file:
                fieldnames = ['center_name', 'm1','s1','n1', 'm2','s2','n2']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

                for em, es, en, cm, cs, cn, center_name in zip(
                    ems, ess, count_egs, cms, css, count_cgs, center_names
                ):
                    writer.writerow({'center_name': center_name,
                                    'm1': em,
                                    's1': es,
                                    'n1': en,
                                    'm2': cm,
                                    's2': cs,
                                    'n2': cn,})

# %%
