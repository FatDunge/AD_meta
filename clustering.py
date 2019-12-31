#%%
import datasets
import numpy as np

csv_prefix = 'csv/{}.csv'

center_edsd = datasets.load_centers_edsd(use_csv=True,
                                         csv_prefix=csv_prefix)
center_mcad = datasets.load_centers_mcad(use_csv=True,
                                         csv_prefix=csv_prefix)
center_list = center_mcad + center_edsd
#%%
persons = []
labels = []
for center in center_list:
    for i in range(3):
        persons += center.get_by_label(i)

datas = []
for person in persons:
    datas.append(person.dataframe.values.flatten())
    labels.append(person.label)
datas = np.array(datas)
labels = np.array(labels)
#%%
from sklearn.preprocessing import MaxAbsScaler
transformer = MaxAbsScaler().fit(datas)
datas = transformer.transform(datas)
#%%
from sklearn.cluster import OPTICS
ms = OPTICS()
ms.fit(datas)
labels_pre = ms.labels_
# %%
from sklearn.decomposition import SparsePCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

pca = SparsePCA(n_components=3)
datas_trans = pca.fit_transform(datas)

def plot_pca_scatter(labels, colors):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig)

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    for i in range(n_clusters_+1):
        px = datas_trans[:, 0][labels==i]
        py = datas_trans[:, 1][labels==i]
        pz = datas_trans[:, 2][labels==i]
        ax.scatter(px, py, pz, c=colors[i])
    plt.show()

colors = ['blue', 'yellow', 'red', 'aqua', 'brown', 'cyan', 'green', 'ivory', 'lime']

plot_pca_scatter(labels, colors)
plot_pca_scatter(labels_pre, colors)

# %%
