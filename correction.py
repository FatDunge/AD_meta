#%%
import os
import numpy as np
import nibabel as nib
from nibabel import nifti1
from scipy.stats import norm

path = './report/ALL'
files = os.listdir(path)

def voxelwise_correction(array, p_array, thres=0.05):
    voxel_count = np.size(p_array)
    thresed_p = p_array < thres / voxel_count
    return np.multiply(array, thresed_p)

for f in files:
    if 'without' in f:
        f_path = os.path.join(path, f)
        niis =  nib.load(f_path)
        arrays = np.asarray(niis.dataobj)

        z_array = arrays[..., -2]
        p_array = norm.sf(z_array) * 2

        for i in range(arrays.shape[-1]):
            array = arrays[..., i]
            corrected_array = voxelwise_correction(array, p_array)
            arrays[..., i] = corrected_array
        affine = niis.affine
        header = niis.header
        corrected_niis = nib.Nifti1Image(arrays, affine, header)
        new_f = f.replace('without', 'with')
        new_path = os.path.join(path, new_f)
        nifti1.save(corrected_niis, new_path)


# %%
arra = np.asarray([[1,2,3,4],[5,6,7,8]])
p_a = np.asarray([[1,0.002, 2, 3], [0.001,2, 0.001, 0.00001]])
voxelwise_correction(arra, p_a)

# %%
