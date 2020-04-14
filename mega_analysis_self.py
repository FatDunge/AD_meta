#%%
import os
import meta_analysis
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
import numpy as np
from meta_analysis.main import voxelwise_meta_analysis
from meta_analysis import utils

nii_prefix = 'mri_smoothed/{}.nii'
filenames_ = ['origin.csv', 'HYDRA.csv']
labels_ = [['NC', 'MC', 'AD'],
          ['NC', 'Subtype_1', 'Subtype_2']]
pairs_ = [[(2,0), (1,0), (2, 1)],
         [(11,9), (12,9)]]
thres = 5
mask_path = './data/mask/grey_matter_smoothed_005.nii'
output = r'./results/meta/{}_{}'

mask_nii = nib.load(mask_path)
mask = Mask(np.asarray(mask_nii.dataobj))
#%%
for filenames, labels, pairs in zip(filenames_, labels_, pairs_):
    centers = datasets.load_centers_all(filenames=filenames)
    for pair in pairs:
        center_dict = {}
        group1_label = pair[0]
        group2_label = pair[1]
        
        out_dir = output.format(group1_label, group2_label)
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
                n1 = len(center.get_by_label(group1_label))
                n2 = len(center.get_by_label(group2_label))
                if n1 < thres or n2 < thres:
                    continue
                print('{}: e:{}, c:{}'.format(center.name, n1, n2), file=text_file)
                group_mean_dict = {}
                group_std_dict = {}
                group_count_dict = {}
                for label in [group1_label, group2_label]:
                    m, s, n = center.load_label_msn(label)
                    group_mean_dict[label] = m
                    group_std_dict[label] = s
                    group_count_dict[label] = n

                center_mean_dict[center.name] = group_mean_dict
                center_std_dict[center.name] = group_std_dict
                center_count_dict[center.name] = group_count_dict

        results = voxelwise_meta_analysis(group1_label, group2_label,
                                        center_mean_dict=center_mean_dict,
                                        center_std_dict=center_std_dict,
                                        center_count_dict=center_count_dict,
                                        _mask=mask, dtype=np.float16)
        
        result_names = ['es','var', 'se', 'll','ul','q','z','p']
        for result, name in zip(results, result_names):
            path = os.path.join(out_dir, '{}.nii'.format(name))
            utils.gen_nii(result, mask_nii, path)

# %%
# Generate mean, std nii
import os
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
from meta_analysis import utils
import numpy as np
nii_prefix = 'mri_smoothed'
filenames = 'HYDRA.csv'
labels = [9, 11, 12]
mask_path = './data/mask/grey_matter_smoothed_005.nii'
centers = datasets.load_centers_all(filenames=filenames)
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
        group = center.get_by_label(label)
        group1_data, _ = center.get_nii_pathes(persons=group, nii_prefix=nii_prefix+'/{}.nii')
        datas = utils.load_arrays(group1_data, dtype=np.float32)
        mean, std, count = utils.cal_mean_std_n(datas)
        mean_path = os.path.join(out_dir, 'mean')
        std_path = os.path.join(out_dir, 'std')
        utils.gen_nii(mean, mask_nii, mean_path)
        utils.gen_nii(std, mask_nii, std_path)

#%%