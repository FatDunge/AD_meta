#%%
import nibabel as nib
import nilearn as nil
from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import combine_pvalues

atlas_path = './data/mask/BN_Atlas_246_1mm_r.nii'
atlas = nib.load(atlas_path)
atlas_array = np.asarray(atlas.dataobj)

nii_path = './report/EDSD/voxelwise_4D_ADNC_with_correction.nii'
nii = nib.load(nii_path)
es_nii = nil.image.index_img(nii, 0)
es_nii_array = np.asarray(es_nii.dataobj)

#%%
for i in range(244, np.max(atlas_array)):
    index = atlas_array == i
    plotting.plot_img(nil.image.new_img_like(atlas, index))

# %%
values = []
for i in range(np.min(atlas_array), np.max(atlas_array)):
    index = atlas_array == i
    count = len(atlas_array[atlas_array==i])
    masked = np.multiply(index, es_nii_array)
    masked_abs = np.absolute(masked)
    value = np.sum(masked_abs) / count
    values.append(value)


# %%
import heapq
max_num_index_list = list(map(values.index, heapq.nlargest(20, values)))

def get_combined(nii_array, atlas_array, max_num_index_list):
    values = []
    for max_num_index in max_num_index_list:
        index = atlas_array == max_num_index
        count = len(atlas_array[atlas_array==max_num_index])
        masked = np.multiply(index, nii_array)
        value = np.sum(masked) / count
        values.append(value)
    return values

def get_p_combined(p_array, atlas_array, max_num_index_list):
    values = []
    for max_num_index in max_num_index_list:
        index = atlas_array == max_num_index
        masked = np.multiply(index, p_array).flatten()
        p_values = masked[masked!=0]
        p_values = 10 ** (-p_values)
        _, p = combine_pvalues(p_values)
        values.append(p)
    return values

#%%
ess = get_combined(es_nii_array, atlas_array, max_num_index_list)
lcis = get_combined(np.asarray(nil.image.index_img(nii, 1).dataobj),
                    atlas_array, max_num_index_list)
ucis = get_combined(np.asarray(nil.image.index_img(nii, 2).dataobj),
                    atlas_array, max_num_index_list)
i_squares = get_combined(np.asarray(nil.image.index_img(nii, 4).dataobj),
                    atlas_array, max_num_index_list)
ses = get_combined(np.asarray(nil.image.index_img(nii, 3).dataobj),
                    atlas_array, max_num_index_list)
ps = get_p_combined(np.asarray(nil.image.index_img(nii, 5).dataobj),
                    atlas_array, max_num_index_list)
info = np.dstack((ess, lcis, ucis, i_squares, ses, ps))[0]

import pandas as pd
df = pd.read_csv('./data/mask/BNA_subregions.csv',index_col=0)
#%%
def draw(info, df, max_num_index_list,gap=0.01, box_size=0.04):
    """
    info:[[es, lci, uci, i^2],[[es, lci, uci, i^2]]]
    """
    n = len(info)
    for i in range(n):
        es, lci, uci, i_square, _, __ = info[i]
        y_low = (n-i-1) * box_size + (n-i)*gap
        y_high = y_low + box_size
        x_box = [es - box_size/2,
                 es - box_size/2,
                 es + box_size/2,
                 es + box_size/2]
        y_box = [y_low,y_high,y_high,y_low]
        if i_square > 50:
            plt.plot(x_box, y_box, color="blue", lw=1)
        else:
            plt.fill(x_box, y_box, color="blue", lw=1)
        
        x_lci = [lci, lci]
        y_lci = [y_low, y_high]
        plt.plot(x_lci, y_lci, color="blue", lw=1)
        x_lci = [uci, uci]
        y_lci = [y_low, y_high]
        plt.plot(x_lci, y_lci, color="blue", lw=1)
        x_line = [lci, uci]
        y_line = [y_low + box_size/2, y_low + box_size/2]
        plt.plot(x_line, y_line, color="blue", lw=1)
        
        text = df.loc[max_num_index_list[i]]['name']
        plt.text(lci-0.2, y_low, text, ha='left', wrap=True)
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

draw(info, df, max_num_index_list)

# %%
import csv
file_path = './report/EDSD/values.csv'
with open(file_path, 'w', newline='') as file:
        fieldnames = ['Region','ES', 'LCI', 'UCI', 'I^2', 'SE', 'P']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(20):
            tmp = info[i]
            writer.writerow({'Region': df.loc[max_num_index_list[i]]['name'],
                            'ES': round(tmp[0], 2),
                            'LCI': round(tmp[1], 2),
                            'UCI': round(tmp[2], 2),
                            'I^2': round(tmp[3], 2),
                            'SE': round(tmp[4], 2),
                            'P': tmp[5]})


# %%
tmp = np.zeros_like(es_nii_array)
for i in range(20):
    index = atlas_array == max_num_index_list[i]
    masked = index * ess[i]
    tmp += masked
new = nil.image.new_img_like(es_nii, tmp)

# %%
nib.nifti1.save(new, './report/tmp.nii')

# %%
plotting.plot_glass_brain(new, colorbar=True, cmap='hot')

# %%
