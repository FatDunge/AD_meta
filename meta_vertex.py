#%%
import os
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
from meta_analysis import utils
import numpy as np
import nilearn as nil
from nilearn.surface import load_surf_data

gii_prefix = 'surf'
filenames = 'origin.csv'
labels = [0, 1, 2]
centers = datasets.load_centers_all(filenames=filenames)
for center in centers:
    for label in labels:
        out_dir = os.path.join(center.file_dir, gii_prefix)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_dir = os.path.join(out_dir, '{}'.format(label))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        pathes, _ = center.get_resampled_gii_pathes(label=label)
        datas = []
        if pathes is not None:
            for path in pathes:
                tmp = load_surf_data(path)
                datas.append(tmp[3])
            mean, std, count = utils.cal_mean_std_n(datas)
            mean_path = os.path.join(out_dir, 'mean')
            std_path = os.path.join(out_dir, 'std')
            np.save(mean_path, mean)
            np.save(std_path, std)

# %%
import os
import meta_analysis
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
import numpy as np
from meta_analysis.main import voxelwise_meta_analysis
from meta_analysis import utils
from nibabel.gifti.gifti import GiftiDataArray,GiftiImage

_filenames = ['origin.csv']
_labels = [['NC', 'MC', 'AD']]
_pairs = [[(2,0), (1,0), (2, 1)]]
mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
output = r'./results/meta/{}_{}'
temp_dir = r'./data/mask/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/{}'
surfs = ['fsaverage.L.inflated.32k_fs_LR.surf.gii', 'fsaverage.R.inflated.32k_fs_LR.surf.gii']
l_r = ['L', 'R']

for filenames, labels, pairs in zip(_filenames, _labels, _pairs):
    centers = datasets.load_centers_all(filenames=filenames)
    for pair in pairs:
        center_dict = {}
        group1_label = pair[0]
        group2_label = pair[1]
        
        out_dir = output.format(group1_label, group2_label)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        out_dir = os.path.join(out_dir, 'surf')
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, 'centers.txt'), "w") as text_file:
            center_mean_dict = {}
            center_std_dict = {}
            center_count_dict = {}
            for center in centers:
                n1 = len(center.get_by_label(group1_label))
                n2 = len(center.get_by_label(group2_label))
                if n1 == 0 or n2 == 0:
                    continue
                print('{}: e:{}, c:{}'.format(center.name, n1, n2), file=text_file)
                group_mean_dict = {}
                group_std_dict = {}
                group_count_dict = {}
                for label in [group1_label, group2_label]:
                    m, s, n = center.load_msn_array(label, _dir='surf')
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
                                        dtype=np.float32)
        
        result_names = ['es','var', 'se', 'll','ul','q','z','p']
        for result, name in zip(results, result_names):
            result_l = result[:32492]
            result_r = result[32492:]
            result_list = [result_l, result_r]

            for _result, surf, lr in zip(result_list, surfs, l_r):
                path = os.path.join(out_dir, '{}_{}.gii'.format(name, lr))
                gii_path = temp_dir.format(surf)
                ct_gii = nib.load(gii_path)
                gdarray = GiftiDataArray.from_array(_result, intent=0)
                ct_gii.remove_gifti_data_array_by_intent(0)
                ct_gii.add_gifti_data_array(gdarray)
                nib.save(ct_gii, path)
