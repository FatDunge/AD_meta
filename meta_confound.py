#%%
import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt
import meta_analysis
import os
import csv
from meta_analysis import utils
from meta_analysis.main import csv_meta_analysis

centers = datasets.load_centers_all()

csv_dir_prefix = './data/meta_csv/{}_{}/confound/'
output_dir_prefix = './results/meta/{}_{}/confound/'
pairs = [(2, 0), (1, 0), (2, 1)]
confounds = ['age', 'gmv', 'csf','wmv', 'tiv']
#%%
# Create csv
confounds = ['gender']
for pair in pairs:
    eg_label = pair[0]
    cg_label = pair[1]
    csv_dir = csv_dir_prefix.format(eg_label, cg_label)
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)
    for confound in confounds:
        csv_path = csv_dir + confound + '.csv'
        with open(csv_path, 'w', newline='') as file:
            #fieldnames = ['center_name', 'm1','s1','n1', 'm2','s2','n2']
            fieldnames = ['center_name', 'a','c', 'b', 'd']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for center in centers:
                """
                if confound == 'age':
                    eg_features, *_ = center.get_ages(eg_label)
                    cg_features, *_ = center.get_ages(cg_label)
                    if eg_features is not None and cg_features is not None:
                        eg_features = eg_features * 100
                        cg_features = cg_features * 100
                elif confound == 'gmv':
                    eg_features, *_ = center.get_gmvs(eg_label)
                    cg_features, *_ = center.get_gmvs(cg_label)
                elif confound == 'csf':
                    eg_features, *_ = center.get_csfs(eg_label)
                    cg_features, *_ = center.get_csfs(cg_label)
                elif confound == 'wmv':
                    eg_features, *_ = center.get_wmvs(eg_label)
                    cg_features, *_ = center.get_wmvs(cg_label)
                elif confound == 'tiv':
                    eg_features, *_ = center.get_tivs(eg_label)
                    cg_features, *_ = center.get_tivs(cg_label)
                if eg_features is not None and cg_features is not None:
                    mean_eg, std_eg, n_eg = utils.cal_mean_std_n(eg_features)
                    mean_cg, std_cg, n_cg = utils.cal_mean_std_n(cg_features)
                    writer.writerow({'center_name': center.name,
                                    'm1': mean_eg,
                                    's1': std_eg,
                                    'n1': n_eg,
                                    'm2': mean_cg,
                                    's2': std_cg,
                                    'n2': n_cg,})
                """
                eg_features, *_ = center.get_genders(eg_label)
                cg_features, *_ = center.get_genders(cg_label)
                if eg_features is not None and cg_features is not None:
                    a = len(eg_features[eg_features==1])
                    c = len(eg_features[eg_features==0])
                    b = len(cg_features[cg_features==1])
                    d = len(cg_features[cg_features==0])

                    writer.writerow({'center_name': center.name,
                                        'a': a,
                                        'c': c,
                                        'b': b,
                                        'd': d,})
# %%
confounds = ['gender']
for pair in pairs:
    eg_label = pair[0]
    cg_label = pair[1]
    csv_dir = csv_dir_prefix.format(eg_label, cg_label)
    output_dir = output_dir_prefix.format(eg_label, cg_label)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for confound in confounds:
        csv_path = csv_dir + confound + '.csv'
        output_path = output_dir + confound + '.png'
        model = csv_meta_analysis(csv_path, method='rr', data_type='cate', model_type='random')
        print(eg_label,cg_label,confound,model.p)
        #model.plot_forest(title=confound, save_path=output_path, show=False)

# %%
