#%%
import nilearn as nil
import nibabel as nib

nii = nib.load('./data/report/mean_z_score_ad_m0wc1_mult.nii')
ref = nib.load('./data/mask/rBN_Atlas_246_1mm.nii')

nii = nil.image.resample_img(nii, ref.affine, ref.get_shape())
nib.save(nii, './data/report/mean_z_score_ad_m0wc1_mult_181.nii')

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nilearn as nil
import nibabel as nib
import mask
from scipy.stats import pearsonr

path = './data/others_result/181'
_mask = mask.Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')
df = pd.read_csv('./report/roi/brainnetome/ALL_ADNC.csv')

filenames = os.listdir(path)

def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
for filename in filenames:
    nii_path = os.path.join(path, filename)
    nii = nib.load(nii_path)

    volumes = _mask.get_all_masked_mean(nii)
    volumes = np.array(volumes)
    es = df['ES'].values
    r, p = pearsonr(volumes, es)
    print('{}: r:{:.2f}, p(two-tailed):{}'.format(filename, r, p))
    volumes_norm = norm(volumes)
    es_norm = norm(es)

    plt.scatter(volumes, es, norm=True)
    plt.xlabel(filename)
    plt.ylabel('Effect size')
    plt.show()

# %%


