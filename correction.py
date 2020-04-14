#%%
import os
import numpy as np
import nibabel as nib
from nibabel import nifti1
from scipy.stats import norm


def voxelwise_correction(array, p_array, voxel_count, thres=0.05):
    thresed_p = p_array < thres / voxel_count
    return np.multiply(array, thresed_p)

mask_path = './data/mask/grey_matter_smoothed_005.nii'
mask_nii = nib.load(mask_path)
mask = np.asarray(mask_nii.dataobj)
voxel_count = np.size(mask[mask!=0])
#%%
path = './results/meta'
tests = os.listdir(path)
for test in tests:
    voxel_path = os.path.join(path, test, 'voxel')
    es_path = os.path.join(voxel_path, 'es.nii')
    p_path =  os.path.join(voxel_path, 'p.nii')
    
    es =  nib.load(es_path)
    es_array = np.asarray(es.dataobj)
    p =  nib.load(p_path)
    p_array = np.asarray(p.dataobj)

    corrected_array = voxelwise_correction(es_array, p_array, voxel_count)
    affine = es.affine
    header = es.header
    corrected_niis = nib.Nifti1Image(corrected_array, affine, header)
    new_f = os.path.join(voxel_path,'es_bon.nii')
    nifti1.save(corrected_niis, new_f)

# %%
