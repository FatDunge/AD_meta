#%%
import os
import numpy as np
import nibabel as nib
from nibabel import nifti1
from scipy.stats import norm

def bonferroni_correction(array, p_array, voxel_count, thres=0.05):
    thresed_p = p_array < thres / voxel_count
    return np.multiply(array, thresed_p)
#%%
# Voxelwise correction
mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
mask_nii = nib.load(mask_path)
mask = np.asarray(mask_nii.dataobj)
voxel_count = np.size(mask[mask!=0])
path = './results/meta'
tests = os.listdir(path)
ps = [0.05, 0.01, 0.001]
for test in tests:
    voxel_path = os.path.join(path, test, 'voxel')
    es_path = os.path.join(voxel_path, 'es.nii')
    p_path =  os.path.join(voxel_path, 'p.nii')
    
    es =  nib.load(es_path)
    es_array = np.asarray(es.dataobj)
    p =  nib.load(p_path)
    p_array = np.asarray(p.dataobj)

    for p in ps:
        corrected_array = voxelwise_correction(es_array, p_array, voxel_count, thres=p)
        affine = es.affine
        header = es.header
        corrected_niis = nib.Nifti1Image(corrected_array, affine, header)
        new_f = os.path.join(voxel_path,'es_bon_{}.nii'.format(str(p)[2:]))
        print(new_f)
        print(len(corrected_array[corrected_array!=0]))
        nifti1.save(corrected_niis, new_f)

# %%
# ROI GMV correction
import numpy as np
from meta_analysis import utils

test_count = 246
path = './results/meta'
tests = os.listdir(path)
ps = [0.05, 0.01, 0.001]
percents = [0.3, 0.5, 1]

def load_nii_array(filepath):
    nii =  nib.load(filepath)
    return np.asarray(nii.dataobj), nii

for test in tests:
    result_path = os.path.join(path, test, 'roi_gmv_removed')
    es_array, es_nii = load_nii_array(os.path.join(result_path, 'es.nii'))
    p_array, _ = load_nii_array(os.path.join(result_path, 'p.nii'))

    for p in ps:
        corrected_array = bonferroni_correction(es_array, p_array, test_count, thres=p)
        for percent in percents:
            unique = np.unique(corrected_array)
            sorted_unique = np.sort(unique)
            n = int(percent*len(sorted_unique))
            thres = sorted_unique[n-1]
            corrected_array[corrected_array > thres] = 0
            cor_path = os.path.join(result_path, 'es_{}_top{}.nii'.format(str(p).replace('0.', ''),
                                                                      int(percent*100)))
            utils.gen_nii(corrected_array, es_nii, cor_path)


# %%
# gii correction
from nilearn.surface import load_surf_data
from nibabel.gifti.gifti import GiftiDataArray
import numpy as np
path = './results/meta'
tests = os.listdir(path)
ps = [0.05, 0.01, 0.001]
l_r = ['L', 'R']
temp_dir = r'./data/mask/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/{}'
surfs = ['fsaverage.L.inflated.32k_fs_LR.surf.gii', 'fsaverage.R.inflated.32k_fs_LR.surf.gii']
for test in tests:
    voxel_path = os.path.join(path, test, 'surf')
    for lr,surf in zip(l_r, surfs):
        es_path = os.path.join(voxel_path, 'es_{}.gii'.format(lr))
        p_path =  os.path.join(voxel_path, 'p_{}.gii'.format(lr))
        
        es_array = load_surf_data(es_path)[-1]
        p_array = load_surf_data(p_path)[-1]

        voxel_count = np.size(p_array) * 2

        for p in ps:
            corrected_array = voxelwise_correction(es_array, p_array, voxel_count, thres=p)
            new_f = os.path.join(voxel_path,'es_bon_{}_{}.gii'.format(lr,str(p)[2:]))
            ct_gii = nib.load(temp_dir.format(surf))
            gdarray = GiftiDataArray.from_array(corrected_array, intent=0)
            ct_gii.add_gifti_data_array(gdarray)
            nib.save(ct_gii, new_f)
        


# %%
