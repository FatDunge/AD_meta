#%%
import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt
import meta_analysis

centers = ['EDSD', 'MCAD']
centers_mcad = datasets.load_centers_mcad(use_nii=False, use_csv=False,
                                          use_personal_info=True, use_xml=True)
centers_edsd = datasets.load_centers_edsd(use_nii=False, use_csv=False,
                                          use_personal_info=True, use_xml=True)

center_list = centers_mcad + centers_edsd
#%%
def get_data_msn(persons, confound='MMSE'):
    if confound == 'MMSE':
        data = np.array([person.get_presonal_info_values()[-1] for person in persons])
    elif confound == 'TIV':
        data = np.array([person.get_tiv() for person in persons])
    elif confound == 'GMV':
        data = np.array([person.get_total_cgw_volumn()[1] for person in persons])
    mean = np.nanmean(data)
    std = np.nanstd(data)
    count = np.count_nonzero(~np.isnan(data))
    return mean, std, count

def gen_study(center, study, label_eg, label_cg, confound):
    mean_eg, std_eg, count_eg = get_data_msn(center.get_by_label(label_eg), confound)
    mean_cg, std_cg, count_cg = get_data_msn(center.get_by_label(label_cg), confound)
    
    if count_eg != 0 and count_cg != 0:
        study.append('{}, {}, {}, {}, {}, {}, {}'.format(center.name,
                                                        mean_eg, std_eg, count_eg,
                                                        mean_cg, std_cg, count_cg))

SETTINGS = {"datatype":"CONT",
            "models":"Random",
            "algorithm":"IV-Cnd",
            "effect":"SMD"}

labels = ['NC', 'MCI', 'AD']
pairs = [(2,1), (1,0),(2,0)]
confounds = ['MMSE', 'TIV', 'GMV']
for confound in confounds:
    for pair in pairs:
        study = []
        for center in center_list:
            gen_study(center, study, pair[0], pair[1], confound)
        results = meta_analysis.meta(study, SETTINGS)
        meta_analysis.show_forest(results, title='{}-{}{}'.format(confound,
                                                                  labels[pair[0]],labels[pair[1]]))

