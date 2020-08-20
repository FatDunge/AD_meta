# %%
# Effect size with MMSE t-value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import seaborn as sns

from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr

import datasets
import utils
from meta_analysis.main import csv_meta_analysis

plt.ioff()

class Result(object):
    def __init__(self, name, r, p):
        self.name = name
        self.r = r
        self.p = p

def cor_roi_confound(roi_models, confound_model, mask,
                     out_dir):
    confound_effect_sizes = confound_model.effect_sizes
    rs = {}
    ps = {}
    for key, roi_model in roi_models.items():
        roi_effect_sizes = roi_model.effect_sizes
        r = pearsonr(confound_effect_sizes, roi_effect_sizes)[0]
        p = pearsonr(confound_effect_sizes, roi_effect_sizes)[1]
        key = int(key)
        rs[key] = r
        ps[key] = p

    nii_array = mask.data.astype(np.float32)
    p_array = mask.data.astype(np.float32)
    ll = [i for i in range(1, 247)]
    for (k, r), (_, p) in zip(rs.items(), ps.items()):
        nii_array[nii_array==k] = r
        p_array[p_array==k] = p
        ll.remove(k)
    for i in ll:
        nii_array[nii_array==np.float32(i)] = 0

    path = os.path.join(out_dir, 'r.nii')
    p_path = os.path.join(out_dir, 'p.nii')
    utils.gen_nii(nii_array, mask.nii, path)
    utils.gen_nii(p_array, mask.nii, p_path)

# Correlation with PET
def cor_roi_pet(roi_models, pet_dir,
                fig_width=5, fig_height=5,
                out_dir=None, show=False, save=True,
                fontsize=14):
    files = os.listdir(pet_dir)
    roi_es_dict = {}
    for k, v in roi_models.items():
        roi_es_dict[k] = v.total_effect_size

    roi_df = pd.DataFrame.from_dict(roi_es_dict, orient='index', columns=['es'])
    roi_df.index.name = 'ID'
    roi_df.index = roi_df.index.map(int)

    results = []
    for f in files:
        path = os.path.join(pet_dir, f)
        df = pd.read_csv(path, index_col=0)
        nndf = pd.merge(roi_df, df, left_on='ID', right_on='ID', left_index=True)

        x = nndf['es'].to_list()
        y = nndf['Volume'].to_list()

        r = pearsonr(x, y)[0]
        p = pearsonr(x, y)[1]
        if 'SERT' in f:
            result = Result(f[:f.rfind('_')], r, p)
        else:
            result = Result(f[:f.find('_')], r, p)
        results.append(result)

        if show or save:
            _, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)))
            ax = sns.regplot(x=x,y=y, robust=True,
                             ax=ax)
            ax.set_title('r={:.2f}, p={:.2e}'.format(r, p), fontdict={'fontsize': fontsize})
            ax.set_xlabel('Effect size of ROIs', fontsize=fontsize)
            ax.set_ylabel(f[:-4], fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            if show:
                plt.show()
            if save:
                plt.savefig(os.path.join(out_dir, f[:-4]+'.png'))
            plt.close()
    return results