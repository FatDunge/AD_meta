#%%
import center
import datasets
import numpy as np
import mask
import nibabel as nib
from sklearn.preprocessing import OneHotEncoder

nii_prefix='mri_smoothed/{}.nii'
_mask = mask.Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')

centers = datasets.load_centers_all()
#%%
def get_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] > item]

def onehot(n_labels, labels):
    return np.eye(n_labels)[labels]

def minmax(x, axis=0):
    _min = np.min(x, axis=axis)
    _max = np.max(x, axis=axis)
    return (x - _min) / (_max - _min)

def z_norm(x, axis=0):
    _mean = np.mean(x, axis=axis)
    _std = np.std(x, axis=axis)
    return (x - _mean) / _std

def create_x(center, intercept=None):
    # age, male, female
    amfms, labels = center.get_presonal_info_values()
    amfs = np.reshape(amfms[:,0:3],(-1,3))
    onehot_lables = onehot(3, labels)
    tcgws, *_ = center.get_tivs_cgws()
    tivs = np.reshape(tcgws[:,0],(-1,1))
    tivs = z_norm(tivs)
    if intercept:
        intercepts = np.reshape(np.repeat(intercept, len(labels)),(-1, 1))
        x = np.hstack((amfs, tivs, intercepts, onehot_lables))
    else:
        x = np.hstack((amfs, tivs, onehot_lables))
    return x

def create_y_nii(center, _mask, nii_prefix='mri_smoothed/{}.nii'):
    y = []
    d = _mask.get_mask_data().flatten()
    index = get_index(d, 0)
    pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
    for path in pathes:
        nii = nib.load(path)
        y.append(np.asarray(nii.dataobj).flatten()[index])
    y = np.array(y)
    return y, index
#%%
#Remove Nii
nii_prefix = 'mri_smoothed/{}.nii'
for center in centers:
    if len(center.persons) > 20:
        x = create_x(center, 1)
        y, index = create_y_nii(center, _mask, nii_prefix)
        x_inv = np.linalg.pinv(x)
        beta = np.dot(x_inv, y)
        beta_a = beta[:4]
        y_hat = np.dot(x[:,:4], beta_a)
        pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
        for (person, path, yi, yi_hat) in zip(center.persons, pathes, y, y_hat):
            onii = nib.load(path)
            header = onii.header
            header.set_data_dtype(np.float32)

            image = np.zeros(shape=(181*217*181))
            for i in range(len(index)):
                image[index[i]] = yi[i] - yi_hat[i]
            image = np.reshape(image, (181, 217, 181))
            nii = nib.Nifti1Image(image, onii.affine, header)
            center.save_nii(person, nii)

#%%
#--------------------------------------------------------------
# remove csv feature
from sklearn.linear_model import LinearRegression
import csv
import os
csv_prefix = 'roi_ct/{}.csv'
out_prefix = 'roi_ct_removed/{}.csv'
for center in centers:
    if len(center.persons) > 20:
        x = create_x(center)
        ys, _, ids = center.get_csv_values(prefix=csv_prefix, flatten=True)
        yst = ys.T
        regs = []
        for y in yst:
            reg = LinearRegression().fit(x, y)
            regs.append(reg)
        for (xi, person, yi) in zip(x, center.persons, ys):
            xi = xi[:4]
            path = os.path.join(center.file_dir, out_prefix.format(person.filename))
            with open(path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['ID', 'CT'])
                writer.writeheader()
                for (yii, reg, _id) in zip(yi, regs, ids):
                    yii_hat = yii - np.dot(xi, reg.coef_[:4])
                    writer.writerow({'ID': _id,
                                     'CT': yii_hat})

# %%
