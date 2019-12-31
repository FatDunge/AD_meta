#%%
import pandas as pd
import nibabel as nib
import numpy as np
from mask import Mask
import os
import copy
import nilearn as nil
from nibabel import nifti1
import matplotlib.pyplot as plt

results_path = './report/roi/brainnetome'
results = os.listdir(results_path)
mask = Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')

#%%
def set_value(data, label, value):
    data[data==label] = value
    return data

for result in results:
    if '.csv' in result:
        df = pd.read_csv(os.path.join(results_path, result), index_col=0)
        mask_data = mask.nii.get_fdata()
        es_image = copy.deepcopy(mask_data)
        lci_image = copy.deepcopy(mask_data)
        uci_image = copy.deepcopy(mask_data)
        se_image = copy.deepcopy(mask_data)
        i2_image = copy.deepcopy(mask_data)
        z_image = copy.deepcopy(mask_data)
        p_image = copy.deepcopy(mask_data)
        sign_image = copy.deepcopy(mask_data)

        affine = mask.nii.affine
        header = mask.nii.header
        header.set_data_dtype(np.float32)

        _min, _max = mask.get_min_max_label()
        for i, row in df.iterrows():
            es_image = set_value(es_image, i, row['ES'])
            lci_image = set_value(lci_image, i, row['LCI'])
            uci_image = set_value(uci_image, i, row['UCI'])
            se_image = set_value(se_image, i, row['SE'])
            i2_image = set_value(i2_image, i, row['I^2'])
            z_image = set_value(z_image, i, row['Z'])
            p_image = set_value(p_image, i, row['P'])
            sign_value = row['Sign']
            if sign_value == '***':
                sign_image = set_value(sign_image, i, 3)
            elif sign_value == '**':
                sign_image = set_value(sign_image, i, 2)
            elif sign_value == '*':
                sign_image = set_value(sign_image, i, 1)
            else:
                sign_image = set_value(sign_image, i, 0)

        es = nib.Nifti1Image(es_image, affine, header)
        lci = nib.Nifti1Image(lci_image, affine, header)
        uci = nib.Nifti1Image(uci_image, affine, header)
        se = nib.Nifti1Image(se_image, affine, header)
        i2 = nib.Nifti1Image(i2_image, affine, header)
        z = nib.Nifti1Image(z_image, affine, header)
        p = nib.Nifti1Image(p_image, affine, header)
        sign = nib.Nifti1Image(sign_image, affine, header)

        nii = nil.image.concat_imgs([es, lci, uci, se, i2, z, p, sign])
        nii_path = os.path.join(results_path, result[:-4]) + '.nii'
        nifti1.save(nii, nii_path)


#%%
def draw(df, title, n=20, gap=0.01, box_size=0.04):
    """
    info:[[es, lci, uci, i^2],[[es, lci, uci, i^2]]]
    """
    for i in range(n):
        value = df.iloc[i]
        es = value['ES']
        lci = value['LCI']
        uci = value['UCI']
        i_square = value['I^2']
        text = value['NAME']

        y_low = (n-i-1) * box_size + (n-i)*gap
        y_high = y_low + box_size
        x_box = [es - box_size/2,
                 es - box_size/2,
                 es + box_size/2,
                 es + box_size/2,
                 es - box_size/2]
        y_box = [y_low,y_high,y_high,y_low,y_low]
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
        plt.text(lci-0.2, y_low, text, ha='left', wrap=True)
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(title)
    plt.show()

for result in results:
    if '.csv' in result:
        df = pd.read_csv(os.path.join(results_path, result), index_col=0)
        df_sorted = df.sort_values(by='ES', ascending=True)
        draw(df_sorted, result[:-4])

# %%
from nilearn import image
from nilearn import plotting
for result in results:
    if '.nii' in result:
        nii_path = os.path.join(results_path, result)
        nii = image.load_img(nii_path)
        es = image.index_img(nii, 0)
        p = image.index_img(nii, 7)
        plotting.plot_stat_map(es, title=result[:-4])
        plotting.plot_roi(p, colorbar=True, cmap='Paired',
                          title=result[:-4])

# %%
print(len(results))

# %%
s = './data/ad/ds/dd'
name = s[s.rfind('/')+1:]
name

# %%
