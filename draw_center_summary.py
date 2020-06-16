#%%
import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt

centers = datasets.load_centers_all()

import csv
csv_path = './data/center_info/count.csv'
with open(csv_path, 'w', newline='') as file:
    fieldnames = ['center', 'NC', 'MC', 'AD']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for center in centers:
        n = len(center.get_by_label(0))
        m = len(center.get_by_label(1))
        a = len(center.get_by_label(2))
        writer.writerow({'center': center.name,
                         'NC': n,'MC':m,'AD':a})

# bar plot
import pandas as pd
df = pd.read_csv('./data/center_info/count.csv', index_col=0)

nc = df['NC']
mci = df['MC']
ad = df['AD']

ind = np.arange(1, len(nc)+1)

p1 = plt.barh(ind, nc)
p2 = plt.barh(ind, mci, left=nc)
p3 = plt.barh(ind, ad, left=nc+mci)

plt.ylabel('Center')
plt.legend((p1[0], p2[0], p3[0]), ('NC', 'MCI', 'AD'))
plt.yticks(ticks=ind, labels=df.index.values)
plt.title('subjects count')
plt.show()

# Pie plot
def get_count(series, study):
    tmp = 0
    for index, value in series.iteritems():
        if study in index:
            tmp += value
    return tmp

nc_adni = get_count(nc, 'ADNI')
nc_mcad = get_count(nc, 'MCAD')
nc_edsd = get_count(nc, 'EDSD')
mc_adni = get_count(mci, 'ADNI')
mc_mcad = get_count(mci, 'MCAD')
mc_edsd = get_count(mci, 'EDSD')
ad_adni = get_count(ad, 'ADNI')
ad_mcad = get_count(ad, 'MCAD')
ad_edsd = get_count(ad, 'EDSD')

adni = [nc_adni, mc_adni, ad_adni]
edsd = [nc_edsd, mc_edsd, ad_edsd]
mcad = [nc_mcad, mc_mcad, ad_mcad]

nc = [nc_adni, nc_edsd, nc_mcad]
mc = [mc_adni, mc_edsd, mc_mcad]
ad = [ad_adni, ad_edsd, ad_mcad]
fig, ax = plt.subplots()

radius = 1.5
size = 0.45
vals = np.array([adni, edsd, mcad])
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1,2,3,5,6,7,9,10,11]))

ax.pie(vals.flatten(), 
       labels=['NC', 'MCI', 'AD']*3,
       autopct='%.2f%%',
       pctdistance=0.85,
       radius=radius, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))
ax.pie(vals.sum(axis=1),
       labels=['ADNI', 'EDSD', 'MCAD'],
       labeldistance=0.7,
       radius=radius-size, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))
plt.show()

print(adni)
print(edsd)
print(mcad)
#%%
def get_all(centers, label):
    tmps = None
    for center in centers:
        # change func by demand
        tmp, _ = center.get_tivs_cgws(label)
        if tmp.size != 0:
            tmp = tmp[:,2]
        #tmp, _ = center.get_ages(label)
        #tmp, _ = center.get_MMSEs(label)
        if tmps is None:
            tmps = tmp
        else:
            if tmp is not None:
                tmps = np.concatenate([tmps, tmp])
    return tmps
# %%
import datasets
import numpy as np
centers_adni = datasets.load_centers_adni()
centers_edsd = datasets.load_centers_edsd()
centers_mcad = datasets.load_centers_mcad()
centers_list = [centers_adni, centers_edsd, centers_mcad]
centers_name = ['ADNI', 'EDSD', 'MCAD']
labels = [0, 1, 2]

for centers, name in zip(centers_list, centers_name):
    for label in labels:
        print(name, label)
        agess = get_all(centers, label)
        mean = np.mean(agess)
        std = np.std(agess)
        print('{:.2f}'.format(mean))
        print('({:.2f})'.format(std))
#%%
for centers in centers_list:
    for label in labels:
        males = get_all(centers, label)
        print(label)
        print(len(males[males==1]))
        print(len(males[males==0]))
# %%
import datasets
import numpy as np
centers = datasets.load_centers_all()

ncs = get_all(centers, 0)
mcs = get_all(centers, 1)
ads = get_all(centers, 2)
# Perform ANOVA
from scipy.stats import f_oneway
f, p = f_oneway(ncs, mcs, ads)
print(f, p)

#%%
print(ncs)
# %%
from scipy.stats import chi2_contingency
obs = [[460, 634, 378],[530, 547,451]]
chi2, p, dof, expected = chi2_contingency(obs)
print(p)

# %%
chi2

# %%
