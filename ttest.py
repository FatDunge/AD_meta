#%%
import datasets
import mask
import numpy as np
from scipy.stats import ttest_ind_from_stats

nii_prefix = 'mri/mwp1{}.nii'
centers_list = datasets.load_centers_mcad(use_nii=True, use_csv=False,
                                          nii_prefix=nii_prefix)

_mask = mask.Mask('./data/mask', 'grey_matter_smoothed_005.nii')
d = _mask.get_mask_data().flatten()
#%%
def get_center_voxel_msn_by_label(center, label):
    persons = center.get_by_label(label)
    data = []
    for person in persons:
        data.append(np.asarray(person.nii.dataobj).flatten())
    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    count = data.shape[0]
    return mean, std, count

def gen_voxel_msn(centers_list, label_eg, label_cg):
    for center in centers_list:
        mean_eg, std_eg, count_eg = get_center_voxel_msn_by_label(center, label_eg)
        mean_cg, std_cg, count_cg = get_center_voxel_msn_by_label(center, label_cg)
        
        if count_eg and count_cg:
            t, p = ttest_ind_from_stats(mean_eg, std_eg, count_eg,
                                     mean_cg, std_cg, count_cg)

    return t

#%%
t = gen_voxel_msn([centers_list[0]], 0, 1)

# %%
import nibabel as nib
def create_t_image(t, nii):
    t = np.nan_to_num(t)
    t_image = np.reshape(t, (181, 217, 181))
    header = nii.header
    header.set_data_dtype(np.float32)
    t_nii = nib.Nifti1Image(t_image, nii.affine, header)
    return t_nii

nii = create_t_image(t, centers_list[0].persons[0].nii)

# %%
nib.nifti1.save(nii, './report/t_origin.nii')

# %%
