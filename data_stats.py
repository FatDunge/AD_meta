#%%
import datasets
import numpy as np
import scipy
centers = ['EDSD', 'MCAD']
labels = ['NC', 'MCI', 'AD']

total_ages = []
total_female = []

for center_name in centers:
    if center_name == 'EDSD':
        centers_list = datasets.load_centers_edsd(use_nii=False, use_csv=False,
                                                  use_personal_info=True)
    elif center_name == 'MCAD':
        centers_list = datasets.load_centers_mcad(use_nii=False, use_csv=False,
                                                  use_personal_info=True)
    for label in labels:
        i = labels.index(label)
        persons = []
        for center in centers_list:
            persons = persons + center.get_by_label(i)
        ages = []
        female = []
        for person in persons:
            pi = person.get_presonal_info_values()
            ages.append(pi[0])
            female.append(pi[2])
        total_ages = total_ages + ages
        total_female = total_female + female
        ages = np.array(ages)
        female = np.array(female)
        print('{}_{}_N:{}'.format(center_name, label, len(persons)))
        print('{}_{}_AGE:{}({})'.format(center_name, label, round(np.mean(ages), 2), round(np.std(ages), 2)))
        print('{}_{}_FEMALE:{}'.format(center_name, label, round(len(female[female==1])/len(female), 2)))
total_ages = np.array(total_ages)
total_female = np.array(total_female)
print('N:{}'.format(len(total_female)))
print('AGE:{}({})'.format(round(np.mean(total_ages), 2), round(np.std(total_ages), 2)))
print('FEMALE:{}'.format(round(len(total_female[total_female==1])/len(total_female), 2)))

# %%
