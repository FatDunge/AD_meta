#%%
import datasets

import nibabel as nib
import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np

import draw_results
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
    meta_roi.meta_ct(label_eg, label_cg, mask=mask, save_gii=False, save_nii=True)
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
    meta_confound.create_csv_for_meta(centers, label_eg, label_cg)

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
    roi_ct_models = meta_roi.meta_ct(label_eg, label_cg, save_gii=False, save_nii=False)

    confound_models = meta_confound.meta_confound(label_eg, label_cg)
    gmv_out_dir = out_dir_prefix.format(label_eg, label_cg, 'gmv', confound)
    ct_out_dir = out_dir_prefix.format(label_eg, label_cg, 'ct', confound)
    confound_model = confound_models[confound]
    correlation.cor_roi_confound(roi_gmv_models, confound_model, mask, gmv_out_dir)
    correlation.cor_roi_confound(roi_ct_models, confound_model, mask, ct_out_dir)
#%%
# abeta PET correlation
from abeta_pet import pet_results
out_dir_prefix = './results/correlation/abeta'
label_pairs = [(2, 0)]
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    roi_gmv_models = meta_roi.meta_gmv(label_eg, label_cg, mask, save_nii=False)
    pet_models = pet_results(label_eg, label_cg)
    es1 = [v.total_effect_size for k,v in roi_gmv_models.items()]
    es2 = [m.total_effect_size for m in pet_models]
    draw_results.plot_correlation(es1, es2, 'Effect sizes of ROI GMV', 'Effect sizes of ROI PET')
#%%
# Correlation with JuSpace PET map
pet_dir = './data/PET/masked_mean'
out_dir_prefix = './results/correlation/{}_{}/{}/PET'
results = {}
labels = ['NC', 'MCI', 'AD']
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

    gmv_result = correlation.cor_roi_pet(roi_gmv_models, pet_dir, out_dir=gmv_out_dir,
                                        fig_width=6, fig_height=5,
                                        fontsize=18, save=True, show=False)
    ct_result = correlation.cor_roi_pet(roi_ct_models, pet_dir, out_dir=ct_out_dir,
                                        fig_width=6, fig_height=5,
                                        fontsize=18,  save=True, show=False)
    results['{}_{}_GMV'.format(labels[label_eg], labels[label_cg])] = gmv_result
    results['{}_{}_CT'.format(labels[label_eg], labels[label_cg])] = ct_result
#%%
draw_results.plot_pet_results(results)
#%%
# PLSR with gene
n_perm_boot = 5000
n_components = 5
out_dir_prefix ='./results/gene/{}_{}'
tmp_out_dir_prefix = './results/tmp'
label_pairs = [(1, 0),(2,1)]
for label_pair in label_pairs:
    label_eg = label_pair[0]
    label_cg = label_pair[1]
    out_dir = out_dir_prefix.format(label_eg, label_cg)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    roi_gmv_models = meta_roi.meta_gmv(label_eg, label_cg, mask, save_nii=False)
    roi_ct_models = meta_roi.meta_ct(label_eg, label_cg, save_gii=False, save_nii=False)

    gmv_plsr = gene_analysis.plsr(roi_gmv_models, n_components=n_components,
                                  n_perm=n_perm_boot, n_boot=n_perm_boot,
                                  out_path=os.path.join(out_dir, 'plsr_gmv.csv'))
    ct_plsr = gene_analysis.plsr(roi_ct_models, n_components=n_components,
                                  n_perm=n_perm_boot, n_boot=n_perm_boot,
                                  out_path=os.path.join(out_dir, 'plsr_ct.csv'))

    with open(os.path.join(out_dir, 'plsr_gmv.pickle'), 'wb') as f:
        pickle.dump(gmv_plsr, f)
    with open(os.path.join(out_dir, 'plsr_ct.pickle'), 'wb') as f:
        pickle.dump(ct_plsr, f)
#%%
# Check plsr model
with open(os.path.join(r'./results/tmp', 'plsr_gmv.pickle'), 'rb') as f:
    gmv_plsr = pickle.load(f)
with open(os.path.join(r'./results/tmp', 'plsr_ct.pickle'), 'rb') as f:
    ct_plsr = pickle.load(f)

# %%
path = './data/gene/zhang/all_genes_convergent_evidence.csv'
cfg_df = pd.read_csv(path, index_col=1)
sns.distplot(cfg_df['Total_evidence'].values, kde_kws={'bw':1}, bins=[0, 1, 2, 3, 4, 5,6])
gmv_plsr ='./results/gene/2_0/plsr_gmv.csv'
df = pd.read_csv(gmv_plsr)
df = df.sort_values(['pls1'], ascending=True)
df_top = df[:500]
df_merge = pd.merge(df_top, cfg_df, left_on='gene_name', right_on='Gene', left_index=True)
sns.distplot(df_merge['Total_evidence'].values, kde_kws={'bw':1}, bins=[0, 1, 2, 3, 4, 5,6])


# %%
# Draw Top n result

plot_gmv_top = False
plot_ct_top = True
if plot_gmv_top:
    main_models = meta_roi.meta_gmv(2, 0, mask, save_nii=False)
    sub_models_list = []
    sub_models1 = meta_roi.meta_gmv(2, 1, mask, save_nii=False)
    sub_models2 = meta_roi.meta_gmv(1, 0, mask, save_nii=False)
    sub_models_list.append(sub_models1)
    sub_models_list.append(sub_models2)
elif plot_ct_top:
    main_models = meta_roi.meta_ct(2, 0, mask, save_nii=False, save_gii=False)
    sub_models_list = []
    sub_models1 = meta_roi.meta_ct(2, 1, mask, save_nii=False, save_gii=False)
    sub_models2 = meta_roi.meta_ct(1, 0, mask, save_nii=False, save_gii=False)
    sub_models_list.append(sub_models1)
    sub_models_list.append(sub_models2)

draw_results.draw_top(main_models, sub_models_list,
            legend_names=['AD-NC', 'AD-MCI', 'MCI-NC'],
            offset=0.2, width_ratio=0.1, height_ratio=0.2,
            linewidth=2, point_size=10, fontsize=14,
            topn=60, box_aspect=None, value_aspect='auto')

#%%
dirr = r'H:\workspace\tesa\results\cesa\100'
out_path = os.path.join(dirr, 'radar.png')
files = os.listdir(dirr)
dfs = []
legend_names = []

for f in files:
    if '.csv' in f:
        dfs.append(pd.read_csv(os.path.join(dirr, f), index_col=0))
        legend_names.append(f[:-4])
draw_results.radar_plot(dfs, col_name='0.05 - adjusted',
                        p_thres=0.001,
                        legend_names=legend_names,
                        legend_loc=(-.1, -.1),
                        save=True,
                        out_path=out_path)

# %%
young_centers = datasets.load_centers_young()
old_centers = datasets.load_centers_all()
roi = 217
labels = (0, 2)
draw_results.plot_roi_aging(young_centers, old_centers, labels, roi)

#%%
draw_results.plot_mmse_cor(centers)