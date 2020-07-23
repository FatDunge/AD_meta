#%%
import datasets

import nibabel as nib
import os
import pickle
import pandas as pd
import seaborn as sns

import gene_analysis
import removing_confound
import meta_roi
import meta_voxel
import meta_vertex
import meta_confound
import correlation
from mask import Mask, NiiMask

#Assume all data is organized

#Load dataset
centers = datasets.load_centers_all()

#Define ROI Mask
mask_dir = './data/mask'
mask_name = 'rBN_Atlas_246_1mm.nii'
mask = NiiMask(mask_dir, mask_name)

label_pairs = [(2,0), (2,1), (1,0)]
#%%
#Create ROI GMV/CT csv file
for center in centers:
    # GMV
    center.create_rgmv_csv(mask)
#%%
for center in centers:
    # CT
    center.create_rct_csv()
#%%
#Remove confound
## ROI
#removing_confound.remove_roi(centers)
removing_confound.remove_roi(centers, csv_prefix='roi_ct/{}.csv',
                                out_prefix='roi_ct_removed/{}.csv')
#%%
## Voxel
centers = centers[:-3]
removing_confound.remove_nii(centers, mask)
#%%
## Vertex
removing_confound.remove_gii(centers)

#%%
# perform meta-analysis
## ROI
csv_prefixs = ['roi_ct_removed']

### Create csv for meta analysis
for csv_prefix in csv_prefixs:
    for label_pair in label_pairs:
        label_eg = label_pair[0]
        label_cg = label_pair[1]
        meta_roi.create_csv_for_meta(centers, label_eg, label_cg, csv_prefix)
#%%
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    # GMV meta-analysis
    meta_roi.meta_gmv(label_eg, label_cg, mask)
    
#%%
# CT meta-analysis
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    meta_roi.meta_ct(label_eg, label_cg)
#%%
## Voxel
### Create center mean std nii file
labels = [0, 1, 2]

for center in centers:
    temp_nii = nib.load('./data/mask/save_temp.nii')
    for label in labels:
        center.create_stat_nii(label, temp_nii)
### perform meta-analysis
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    meta_voxel.meta_nii(centers, label_eg, label_cg)

## Vertex
### Create center mean std numpy file

for center in centers:
    for label in labels:
        center.create_stat_gii(label)
#%%
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    meta_vertex.meta_gii(centers, label_eg, label_cg)
#%%
## Confound
### Create csv for meta analysis
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    meta_confound.create_csv_for_meta(centers, label_eg, label_cg, csv_prefix)

for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    meta_confound.meta_confound(label_eg, label_cg)
# %%
# Correlation with MMSE
out_dir_prefix = './results/correlation/{}_{}/{}/{}'
confound = 'MMSE'
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    roi_gmv_models = meta_roi.meta_gmv(label_eg, label_cg, mask, save_nii=False)
    roi_ct_models = meta_roi.meta_ct(label_eg, label_cg, save_gii=False)

    confound_models = meta_confound.meta_confound(label_eg, label_cg)
    gmv_out_dir = out_dir_prefix.format(label_eg, label_cg, 'gmv', confound)
    ct_out_dir = out_dir_prefix.format(label_eg, label_cg, 'ct', confound)
    confound_model = confound_models[confound]
    correlation.cor_roi_confound(roi_gmv_models, confound_model, mask, gmv_out_dir)
    correlation.cor_roi_confound(roi_ct_models, confound_model, mask, ct_out_dir)
#%%
# Correlation with JuSpace PET map
pet_dir = './data/PET/masked_mean'
out_dir_prefix = './results/correlation/{}_{}/{}/PET'
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    roi_gmv_models = meta_roi.meta_gmv(label_eg, label_cg, mask, save_nii=False)
    roi_ct_models = meta_roi.meta_ct(label_eg, label_cg, save_gii=False)

    gmv_out_dir = out_dir_prefix.format(label_eg, label_cg, 'gmv')
    ct_out_dir = out_dir_prefix.format(label_eg, label_cg, 'ct')
    if not os.path.exists(gmv_out_dir):
        os.mkdir(gmv_out_dir)
    if not os.path.exists(ct_out_dir):
        os.mkdir(ct_out_dir)

    correlation.cor_roi_pet(roi_gmv_models, pet_dir, gmv_out_dir)
    correlation.cor_roi_pet(roi_ct_models, pet_dir, ct_out_dir)

#%%
# PLSR with gene
label_pairs = [(2,0)]
out_dir_prefix = './results/gene/{}_{}'
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    out_dir = out_dir_prefix.format(label_eg, label_cg)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    roi_gmv_models = meta_roi.meta_gmv(label_eg, label_cg, mask, save_nii=False)
    roi_ct_models = meta_roi.meta_ct(label_eg, label_cg, save_gii=False)

    gmv_plsr = gene_analysis.plsr(roi_gmv_models, n_components=3,
                                  n_perm=500, n_boot=500,
                                  out_path=os.path.join(out_dir, 'plsr_gmv.csv'))
    ct_plsr = gene_analysis.plsr(roi_ct_models, n_components=3,
                                  n_perm=500, n_boot=500,
                                  out_path=os.path.join(out_dir, 'plsr_ct.csv'))

    with open(os.path.join(out_dir, 'plsr_gmv.pickle'), 'wb') as f:
        pickle.dump(gmv_plsr, f)
    with open(os.path.join(out_dir, 'plsr_ct.pickle'), 'wb') as f:
        pickle.dump(ct_plsr, f)

# %%
path = './data/gene/zhang/all_genes_convergent_evidence.csv'
cfg_df = pd.read_csv(path, index_col=1)
#gene_analysis.cfg_distribution(df['Total_evidence'].values)
sns.distplot(cfg_df['Total_evidence'].values, kde_kws={'bw':1}, bins=[0, 1, 2, 3, 4, 5,6])
gmv_plsr ='./results/gene/2_0/plsr_gmv.csv'
df = pd.read_csv(gmv_plsr)
df = df.sort_values(['pls1'], ascending=False)
df_top = df[:500]
df_merge = pd.merge(df_top, cfg_df, left_on='gene_name', right_on='Gene', left_index=True)
sns.distplot(df_merge['Total_evidence'].values, kde_kws={'bw':1}, bins=[0, 1, 2, 3, 4, 5,6])
# %%
ct_plsr.varexp

# %%
