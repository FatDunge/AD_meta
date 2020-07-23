# %%
# Effect size with MMSE t-value
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import datasets
import os
from meta_analysis.main import csv_meta_analysis
import utils
import nibabel as nib
import seaborn as sns

class Result(object):
    def __init__(self, r, p):
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
                out_dir):
    files = os.listdir(pet_dir)
    roi_es_dict = {}
    for k, v in roi_models.items():
        roi_es_dict[k] = v.total_effect_size

    roi_df = pd.DataFrame.from_dict(roi_es_dict, orient='index', columns=['es'])
    roi_df.index.name = 'ID'
    roi_df.index = roi_df.index.map(int)

    for f in files:
        path = os.path.join(pet_dir, f)
        df = pd.read_csv(path, index_col=0)
        roi_df.index = roi_df.index.map(int)
        nndf = pd.merge(roi_df, df, left_on='ID', right_on='ID', left_index=True)

        x = nndf['es'].to_list()
        y = nndf['Volume'].to_list()

        r = pearsonr(x, y)[0]
        p = pearsonr(x, y)[1]

        sns.regplot(x=x,y=y, robust=True, label='r={:.2f}, p={:.2e}'.format(r, p))
        plt.xlabel('Effect size of ROI')
        plt.ylabel(f[:-4])
        plt.xticks()
        plt.yticks()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f[:-4]+'.png'))
        plt.show()

#%%
"""
# Partial Correlation with PET
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from pingouin import partial_corr

roi_id_path = './data/mask/gmv_id.csv'
roi_id_df = pd.read_csv(roi_id_path, index_col=1)

es_path = './results/meta/2_0/roi_gmv_removed/TOP246.csv'
es_df = pd.read_csv(es_path, index_col=0)

ndf = pd.merge(es_df['es'], roi_id_df['ID'], on='name')
ndf.set_index('ID')

cov_path = './data/mask/tpm.csv'
cov_df = pd.read_csv(cov_path, index_col=0)

pet_path = './data/PET/masked_mean'
files = os.listdir(pet_path)
for f in files:
    path = os.path.join(pet_path, f)
    df = pd.read_csv(path, index_col=0)
    nndf = pd.merge(ndf, df, left_on='ID', right_on='id', left_index=True)
    nndf = pd.merge(nndf, cov_df, left_on='id', right_on='id', left_index=True)
    ans = partial_corr(nndf, x='es', y='value', covar='tpm')
    print(f,ans.iloc[0]['r'])
# %%
ans
"""