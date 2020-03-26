#%%
import meta_analysis
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
import numpy as np
from meta_analysis.main import voxelwise_meta_analysis
from meta_analysis import utils

nii_prefix = 'mri_smoothed/{}.nii'
filenames = 'HYDRA.csv'
centers_mcad = datasets.load_centers_mcad(use_nii=True, use_csv=False,
                                          filenames=filenames, nii_prefix=nii_prefix)
centers_edsd = datasets.load_centers_edsd(use_nii=True, use_csv=False,
                                          filenames=filenames, nii_prefix=nii_prefix)
centers_adni = datasets.load_centers_adni(use_nii=True, use_csv=False,
                                          filenames=filenames, nii_prefix=nii_prefix)
#%%
centers = ['ALL']
labels = ['NC', 'MC', 'AD-1', 'AD-2']
pairs = [(2,0), (3, 0), (3, 2)]
centers_list = centers_edsd + centers_mcad + centers_adni
mask_path = './data/mask/grey_matter_smoothed_005.nii'
mask_nii = nib.load(mask_path)
mask = Mask(np.asarray(mask_nii.dataobj))
#%%
for pair in pairs:
    center_dict = {}
    group1_label = pair[0]
    group2_label = pair[1]
    for center in centers_list:
        group1 = center.get_by_label(group1_label)
        group2 = center.get_by_label(group2_label)
        if len(group1) < 5 or len(group2) < 5:
            continue

        group1_data = [person.get_nii_path() for person in group1]
        group2_data = [person.get_nii_path() for person in group2]
        group_dict = {group1_label: group1_data, group2_label: group2_data}
        center_dict[center.name] = group_dict

    results = voxelwise_meta_analysis(center_dict, group1_label, group2_label, _mask=mask, is_filepath=False)

    result_names = ['es','var', 'se', 'll','ul','q','z','p']
    output = r'./report/my_meta/{}{}_{}'
    for result, name in zip(results, result_names):
        path = output.format(labels[group1_label], labels[group2_label], name)
        utils.gen_nii(result, mask_nii, path)
#%%
np.asarray(np.asarray(centers_list[0].persons[0].nii.dataobj), dtype=np.float16)
# %%
