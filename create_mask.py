#%%
import datasets
import numpy as np

centers_list1 = datasets.load_centers_edsd(use_nii=True, use_csv=False,
                                           nii_prefix='mri_smoothed/{}.nii')
centers_list2 = datasets.load_centers_mcad(use_nii=True, use_csv=False,
                                           nii_prefix='mri_smoothed/{}.nii')
centers_list = centers_list1 + centers_list2
#%%
mask = np.zeros(shape=(181,217,181))

def create_mask(mask, centers_list, threshold):
    n = 0
    for center in centers_list:
        for person in center.persons:
            n += 1
            data = np.asarray(person.nii.dataobj)
            data[data>threshold] = 1
            mask += data
    return mask, n
#%%
mask, n = create_mask(mask, centers_list, 0.05)
tmp = mask
#%%
print(n)
print(len(tmp[tmp>0]))
print(len(tmp[tmp>=n-1]))

#%%
tmp[tmp<n-1]=0
tmp[tmp>=n-1]=1

#%%
import nibabel as nib
person = centers_list[0].persons[0]
nii = nib.Nifti1Image(tmp, 
                      person.nii.affine, person.nii.header)
nib.nifti1.save(nii, './data/mask/grey_matter_smoothed_005_n-1.nii')

#%%
