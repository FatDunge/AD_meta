#%%
import os
import meta_analysis
import datasets
import nibabel as nib
import numpy as np
from meta_analysis.main import voxelwise_meta_analysis
from meta_analysis.mask import Mask
import utils

#%%
# load nii of mean, std, preform voxelwise_meta_analysis
def meta_nii(centers, label_eg, label_cg,
               mask_path='./data/mask/rBN_Atlas_246_1mm.nii',
               mri_dir='mri_smoothed_removed',
               out_dir='./results/meta/{}_{}'):
    mask_nii = nib.load(mask_path)
    mask = Mask(np.asarray(mask_nii.dataobj))
    
    out_dir = out_dir.format(label_eg, label_cg)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = os.path.join(out_dir, 'voxel')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'centers.txt'), "w") as text_file:
        center_mean_dict = {}
        center_std_dict = {}
        center_count_dict = {}
        for center in centers:
            n1 = len(center.get_by_label(label_eg))
            n2 = len(center.get_by_label(label_cg))
            if n1 == 0 or n2 == 0:
                continue
            print('{}: e:{}, c:{}'.format(center.name, n1, n2), file=text_file)
            group_mean_dict = {}
            group_std_dict = {}
            group_count_dict = {}
            for label in [label_eg, label_cg]:
                m, s, n = center.load_msn_nii(label, _dir=mri_dir)
                group_mean_dict[label] = m
                group_std_dict[label] = s
                group_count_dict[label] = n

            center_mean_dict[center.name] = group_mean_dict
            center_std_dict[center.name] = group_std_dict
            center_count_dict[center.name] = group_count_dict

    results = voxelwise_meta_analysis(label_eg, label_cg,
                                    center_mean_dict=center_mean_dict,
                                    center_std_dict=center_std_dict,
                                    center_count_dict=center_count_dict,
                                    _mask=mask, dtype=np.float32)
    
    result_names = ['es','var', 'se', 'll','ul','q','z','p']
    for result, name in zip(results, result_names):
        path = os.path.join(out_dir, '{}.nii'.format(name))
        utils.gen_nii(result, mask_nii, path)

# %%
# Generate mean, std nii
"""
import os
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
from meta_analysis import utils
import numpy as np
nii_prefix = 'mri_smoothed_removed'
filenames = 'origin.csv'
labels = [0, 1, 2]
mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
centers = datasets.load_centers_adni(filenames=filenames)
mask_nii = nib.load(mask_path)
mask = Mask(np.asarray(mask_nii.dataobj))
for center in centers:
    for label in labels:
        out_dir = os.path.join(center.file_dir, nii_prefix)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_dir = os.path.join(out_dir, '{}'.format(label))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        group1_pathes, _ = center.get_nii_pathes(label=label, nii_prefix=nii_prefix+'/{}.nii')
        datas = utils.load_arrays(group1_pathes, dtype=np.float16)
        mean, std, count = utils.cal_mean_std_n(datas)
        mean_path = os.path.join(out_dir, 'mean')
        std_path = os.path.join(out_dir, 'std')
        utils.gen_nii(mean, mask_nii, mean_path)
        utils.gen_nii(std, mask_nii, std_path)

#%%
# Generate mean, std nii
import os
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
from meta_analysis import utils
import numpy as np
nii_prefix = 'mri_smoothed_removed'
filenames = 'origin.csv'
labels = [0, 1, 2]
mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
centers = datasets.load_centers_adni(filenames=filenames)
mask_nii = nib.load(mask_path)

mask = np.ones(shape=(181*217*181))

def get_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] > item]

indexss = get_index(mask, 0)
batch_size = 500000

for center in centers:
    print(center.file_dir)
    for label in labels:
        print(label)
        current = 0
        end = 0
        mean = np.zeros(shape=(181*217*181))
        std = np.zeros(shape=(181*217*181))
        out_dir = os.path.join(center.file_dir, nii_prefix)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_dir = os.path.join(out_dir, '{}'.format(label))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        group_pathes, _ = center.get_nii_pathes(label=label, nii_prefix=nii_prefix+'/{}.nii')
            
        while end < len(indexss):
            print(current)
            end = current + batch_size
            if end > len(indexss):
                end = len(indexss)
            indexs = indexss[current:end]
            current = end

            lst = []
            for path in group_pathes:
                onii = nib.load(path)
                lst.append(np.asarray(onii.dataobj).flatten()[indexs])
 
            ms,ss,n = utils.cal_mean_std_n(lst)
            for index, m, s, in zip(indexs, ms, ss):
                mean[index] = m
                std[index] = s

        mean = np.reshape(mean, newshape=(181,217,181))
        std = np.reshape(std, newshape=(181,217,181))
        mean_path = os.path.join(out_dir, 'mean')
        std_path = os.path.join(out_dir, 'std')
        utils.gen_nii(mean, mask_nii, mean_path)
        utils.gen_nii(std, mask_nii, std_path)

# %%
"""