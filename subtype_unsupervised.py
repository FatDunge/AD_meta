#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
import datasets

feature_path = './matlab/HYDRA/data/features.csv'

df = pd.read_csv(feature_path, index_col=0)
# take only AD group
ad_df = df[df.group == 1]
ad_df = ad_df.drop('group', axis=1)
X = ad_df.values
# norm features
min_max_scaler = preprocessing.MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)
n_clusters = 2

def save_clusters(y, offset, filename):
    i = 0
    y = y + offset
    center_list = datasets.load_centers_all()
    for center in center_list:
        for person in center.persons:
            if person.label == 2:
                person.label = y[i]
                i += 1
    for center in center_list:
        center.save_labels(filename)
# %%
# NMF
from sklearn.decomposition import NMF
model = NMF(n_components=n_clusters)
W = model.fit_transform(X_norm.T)
H = model.components_

y_NMF = np.argmax(H, axis=0)
save_clusters(y_NMF, 20, 'NMF.csv')

#%%
from sklearn.cluster import AgglomerativeClustering
y_ward = AgglomerativeClustering(n_clusters=n_clusters,
        linkage='ward').fit_predict(X_norm)
save_clusters(y_ward, 30, 'ward.csv')

#%%
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(y_NMF, y_ward)

# %%
import datasets

centers = datasets.load_centers_all()
features = []
for center in centers:
    persons = center.get_by_label(2)
    f = center.get_cortical_thickness(persons)
    features.append(f)

# %%
from nibabel.freesurfer.io import read_morph_data
from nilearn.surface import load_surf_data
p = r'E:\workspace\AD_meta\data\AD\ADNI\003\surf\rh.thickness.003_S_0907_NC_ADNI1_screening'
a = read_morph_data(p)

pg = r'E:\workspace\AD_meta\data\AD\EDSD\FRE\surf\lh.sphere.FRE_MCI001.gii'
b = load_surf_data(pg)
# %%
import datasets
import os

centers = datasets.load_centers_all()
for center in centers:
    surf_dir = center.file_dir+'\surf'
    files = os.listdir(surf_dir)
    for f in files:
        if 'lh.thickness' in f: 
            a = read_morph_data(os.path.join(surf_dir, f))
            print(a.shape)
# %%
f = centers[0].get_cortical_thickness()

# %%
import datasets
import numpy as np

centers = datasets.load_centers_all()
features_all = None
for center in centers:
    person_ad = center.get_by_label(2)
    if person_ad:
        features, _ = center.get_cortical_thickness(person_ad)
        if features_all is None:
            features_all = features
        else:
            print(center.name, features.shape)
            features_all = np.concatenate((features, features_all),axis=0)
#%%
features_all = np.nan_to_num(features_all)
# %%
from sklearn.decomposition import NMF
model = NMF(n_components=3)
W = model.fit_transform(features_all.T)
H = model.components_

y_NMF = np.argmax(H, axis=0)
#save_clusters(y_NMF, 20, 'NMF.csv')

# %%
import matplotlib.pyplot as plt

plt.imshow(np.repeat(H,100,axis=0), interpolation='nearest')
plt.show()

# %%
len(y_NMF[y_NMF==0])

# %%
