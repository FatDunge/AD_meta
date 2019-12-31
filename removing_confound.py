#%%
import center
import datasets
import numpy as np
import mask
import nibabel as nib

dataset = 'ADNI'

def get_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] > item]

def create_xy(center, _mask):
    x = []
    y = []
    d = _mask.get_mask_data().flatten()
    index = get_index(d, 0)
    for person in center.persons:
        personal_info = person.get_presonal_info_values()[0:3]
        tiv = person.get_tiv()
        intercept = 1
        label = person.get_label_binary()
        x.append(np.hstack((personal_info, tiv, intercept, label)))
        y.append(np.asarray(person.nii.dataobj).flatten()[index])
    x = np.array(x)
    y = np.array(y)
    return x, y, index

_mask = mask.Mask('./data/mask', 'grey_matter_smoothed_005.nii')
d = _mask.get_mask_data().flatten()
index = get_index(d, 0)

if dataset == 'MCAD':
    centers = datasets.load_centers_mcad(use_nii=True,
                                         use_csv=False,
                                         use_personal_info=True,
                                         use_xml=True,
                                         nii_prefix='mri_smoothed/{}.nii')
elif dataset == 'EDSD':
    centers = datasets.load_centers_edsd(use_nii=True,
                                         use_csv=False,
                                         use_personal_info=True,
                                         use_xml=True,
                                         nii_prefix='mri_smoothed/{}.nii')
elif dataset == 'ADNI':
    centers = datasets.load_centers_adni(use_nii=True,
                                         use_csv=False,
                                         use_personal_info=True,
                                         use_xml=True,
                                         nii_prefix='mri_smoothed/{}.nii')
#%%
i = 0
for center in centers:
    x, y, index = create_xy(center, _mask)
    x_inv = np.linalg.pinv(x)
    beta = np.dot(x_inv, y)
    np.save('./npy/{}_{}_beta.npy'.format(dataset, i), beta)
    i += 1

#%%
j = 0
for center in centers:
    beta = np.load('./npy/{}_{}_beta.npy'.format(dataset, j))
    beta_a = beta[:4]
    for person in center.persons:
        onii = person.nii
        header = onii.header
        header.set_data_dtype(np.float32)

        image = np.zeros(shape=(181*217*181))
        personal_info = person.get_presonal_info_values()[0:3]
        tiv = person.get_tiv()
        x = np.hstack((personal_info, tiv))
        y = np.asarray(person.nii.dataobj).flatten()[index]
        y_hat = np.dot(x, beta_a)
        for i in range(len(index)):
            image[index[i]] = y[i] - y_hat[i]
        image = np.reshape(image, (181, 217, 181))
        nii = nib.Nifti1Image(image, onii.affine, header)
        person.save_image(nii, 'mri_smoothed_removed/{}.nii')
    j += 1

#%%
