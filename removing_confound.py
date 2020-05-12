#%%
import center
import datasets
import numpy as np
import mask
import nibabel as nib
from sklearn.preprocessing import OneHotEncoder

_mask = mask.Mask('./data/mask', 'rBN_Atlas_246_1mm.nii')
centers = datasets.load_centers_all()

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
        y.append(np.asarray(nii.dataobj, dtype=np.float16).flatten()[index])
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
        y_hats = np.dot(x[:,:4], beta_a)
        pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
        for (person, path, yi, yi_hat) in zip(center.persons, pathes, y, y_hats):
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
from sklearn.linear_model import HuberRegressor
import time
nii_prefix = 'mri_smoothed/{}.nii'

d = _mask.get_mask_data().flatten()
indexss = get_index(d, 0)
i = 0
batch_size = 100000
current = 0
end = 0
regs = []
st = time.time()

x = None
print('loading x')
for center in centers:
    if x is None:
        x = create_x(center)
    else:
        x = np.concatenate((x, create_x(center))) 
et = time.time()
print('x time: {}'.format(et-st))
st = et

while end < len(indexss):
    end = current + batch_size
    if end > len(indexss):
        end = len(indexss)
    indexs = indexss[current:end]
    current = end

    ys = []
    for center in centers:
        print('loading y')
        pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
        for path in pathes:
            nii = nib.load(path)
            ys.append(np.asarray(nii.dataobj).flatten()[indexs])
        et = time.time()
        print('y time: {}'.format(et-st))
        st = et
    ys = np.asarray(ys)
    print('Regression')
    for y in ys.T:
        reg = HuberRegressor().fit(x, y)
        regs.append(reg)
    et = time.time()
    print('Regression time:{}'.format(et-st))
    st = et
#%%
import pickle
output = open('./data/regs.pkl', 'wb')
pickle.dump(regs, output)
output.close()

#%%
import pickle
tmp = [reg.coef_ for reg in regs]
output = open('./data/regs_coef.pkl', 'wb')
pickle.dump(tmp, output)
output.close()
#%%
tmp = np.asarray(tmp)
tmp[tmp==0]
#%%
for center in centers:
    pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
    x = create_x(center)
    for (path, xi, person) in zip(pathes, x, center.persons):
        onii = nib.load(path)
        print(path)
        header = onii.header
        header.set_data_dtype(np.float32)
        image = np.zeros(shape=(181*217*181))
        datas = np.asarray(onii.dataobj).flatten()[indexss]
        for v, index, reg in zip(datas, indexss, regs):
            image[index] = v - np.dot(xi[:4], reg.coef_[:4])
        image = np.reshape(image, (181, 217, 181))
        nii = nib.Nifti1Image(image, onii.affine, header)
        center.save_nii(person, nii)
    
#%%
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
# remove csv feature by center
from sklearn.linear_model import HuberRegressor
import csv
import os
csv_prefix = 'roi_gmv/{}.csv'
out_prefix = 'roi_gmv_removed/{}.csv'
for center in centers:
    if len(center.persons) > 20:
        x = create_x(center)
        ys, _, ids = center.get_csv_values(prefix=csv_prefix, flatten=True)
        yst = ys.T
        regs = []
        for y in yst:
            reg = HuberRegressor().fit(x, y)
            regs.append(reg)
        for (xi, person, yi) in zip(x, center.persons, ys):
            xi = xi[:4]
            path = os.path.join(center.file_dir, out_prefix.format(person.filename))
            with open(path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['ID', 'GMV'])
                writer.writeheader()
                for (yii, reg, _id) in zip(yi, regs, ids):
                    yii_hat = yii - np.dot(xi, reg.coef_[:4])
                    writer.writerow({'ID': _id,
                                     'GMV': yii_hat})


# %%
from sklearn.linear_model import HuberRegressor
import csv
import os
csv_prefix = 'roi_ct/{}.csv'
out_prefix = 'roi_ct_removed/{}.csv'

x = None
print('loading x')
for center in centers:
    if x is None:
        x = create_x(center)
    else:
        x = np.concatenate((x, create_x(center))) 
yss = None
for center in centers:
    ys, _, ids = center.get_csv_values(prefix=csv_prefix, flatten=True)
    if yss is None:
        yss = ys
    else:
        yss = np.concatenate((yss, ys)) 
yst = yss.T
regs = []
for y in yst:
    reg = HuberRegressor().fit(x, y)
    regs.append(reg)

for center in centers:
    if len(center.persons) > 20:
        x = create_x(center)
        ys, _, ids = center.get_csv_values(prefix=csv_prefix, flatten=True)
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
