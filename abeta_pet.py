#%%
# process abeta data
from scipy.io import loadmat
import numpy as np
import pandas as pd
from meta_analysis import model, data, utils

def load_data():
    data_mat = loadmat('./data/PET/abeta/pet_value.mat')
    datas = data_mat['pet_value'].T
    return datas

def load_info():
    info_mat = loadmat('./data/PET/abeta/info_corr.mat')

    olabels = info_mat['info'][0][0][-2].flatten()
    labels = []
    for l in olabels:
        if l:
            if l[0] == 'CN':
                labels.append(0)
            elif l[0] == 'MCI':
                labels.append(1)
            elif l[0] == 'Dementia':
                labels.append(2)
        else:
            labels.append(-1)
    return labels

def get_phases():
    name_mat = loadmat('./data/PET/abeta/subname.mat')
    names = name_mat['subname'].flatten()
    center_info = pd.read_csv('./data/center_info/ADNI/final_screening.csv', index_col=2)
    center_labels = []
    for name in names:
        phase = center_info.loc[name[0]]['COLPROT']
        center_labels.append(phase)
    return center_labels

def get_sites():
    name_mat = loadmat('./data/PET/abeta/subname.mat')
    names = name_mat['subname'].flatten()
    sites = []
    for name in names:
        sites.append(name[0][:3])
    return sites

def get_roi_datas(roi, datas, label, labels, phase, all_phases):
    ds = []
    ad = datas[roi]
    for d, l, c in zip(ad, labels, all_phases):
        if l == label and c == phase:
            ds.append(d)
    return ds

def pet_results(eg_label = 2, cg_label = 0):
    all_phases = get_sites()
    phases = np.unique(all_phases)
    result_models = []
    datas = load_data()
    labels = load_info()
    for i in range(246):
        studies = []
        for phase in phases:
            eg_roi_datas = get_roi_datas(i, datas, eg_label, labels, phase, all_phases)
            cg_roi_datas = get_roi_datas(i, datas, cg_label, labels, phase, all_phases)
            if len(eg_roi_datas) > 5 and len(cg_roi_datas) > 5:
                m1, s1, n1 = utils.cal_mean_std_n(eg_roi_datas)
                m2, s2, n2 = utils.cal_mean_std_n(cg_roi_datas)
                group1 = data.NumericalGroup(eg_label, mean=m1, std=s1, count=n1)
                group2 = data.NumericalGroup(cg_label, mean=m2, std=s2, count=n2)
                center = data.Center(phase, [group1, group2])
                studies.append(center.gen_study(eg_label, cg_label, 'cohen_d'))
        result_model = model.RandomModel(studies)
        result_models.append(result_model)
    return result_models
# %%
