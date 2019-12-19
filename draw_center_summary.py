#%%
import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt

centers = ['EDSD', 'MCAD']
centers_mcad = datasets.load_centers_mcad(use_nii=False, use_csv=False,
                                                  use_personal_info=True)
centers_edsd = datasets.load_centers_edsd(use_nii=False, use_csv=False,
                                                  use_personal_info=True)
#%%
ncs = []
mcis = []
ads = []
for center_name in centers:
    nc = []
    mci = []
    ad = []
    if center_name == 'EDSD':
        center_list = centers_edsd
    elif center_name == 'MCAD':
        center_list = centers_mcad
    else:
        center_list = centers_mcad + centers_edsd
    
    for center in center_list:
        nc.append(len(center.get_by_label(0)))
        mci.append(len(center.get_by_label(1)))
        ad.append(len(center.get_by_label(2)))

    nc = np.array(nc)
    mci = np.array(mci)
    ad = np.array(ad)

    ncs.append(nc)
    mcis.append(mci)
    ads.append(ad)

    ind = np.arange(1, len(center_list)+1)

    p1 = plt.bar(ind, nc)
    p2 = plt.bar(ind, mci, bottom=nc)
    p3 = plt.bar(ind, ad, bottom=nc+mci)

    plt.ylabel('Count')
    plt.legend((p1[0], p2[0], p3[0]), ('NC', 'MCI', 'AD'))
    plt.xticks([])
    plt.title('{} center subjects count'.format(center_name))
    plt.show()

# %%
for i in range(len(ncs)):
    ncs[i] = np.sum(ncs[i])
for i in range(len(mcis)):
    mcis[i] = np.sum(mcis[i])
for i in range(len(ads)):
    ads[i] = np.sum(ads[i])
#%%
fig, ax = plt.subplots()

size = 0.3
vals = np.array([ncs, mcis, ads])

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))


plt.show()
# %%