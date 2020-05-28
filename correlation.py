#%%
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
gmvtoct_path = './data/mask/BNA_GMVtoCT.csv'
gmv_path = './results/meta/{}_{}/roi_gmv_removed/Top246.csv'
ct_path = './results/meta/{}_{}/roi_ct_removed/Top210.csv'

pairs = [(1,0)]

gmvtoct = pd.read_csv(gmvtoct_path, index_col=1)
for pair in pairs:
    _gmv_path = gmv_path.format(pair[0], pair[1])
    _ct_path = ct_path.format(pair[0], pair[1])
    gmv = pd.read_csv(_gmv_path, index_col=0)
    ct = pd.read_csv(_ct_path, index_col=0)
    a = []
    b = []
    c = []
    d = []
    for index, row in ct.iterrows():
        es_ct = row['es']
        p = row['p']
        gmv_name = gmvtoct.loc[index]
        if p < 0.001/ 210:
            c.append(gmv_name)
        else:
            d.append(gmv_name)
        gm_row = gmv.loc[gmv_name]
        es_gmv = gm_row['es'].values[0]
        a.append(es_ct)
        b.append(es_gmv)

    a = np.asarray(a)
    b = np.asarray(b)
    r = pearsonr(a, b)[0]
    p = pearsonr(a, b)[1]
    bias, m = polyfit(a, b, 1)

    plt.plot(a, b, '.')
    plt.plot(a, bias + m * a, 
            label='r={:.2f}, p={:.2e}'.format(r, p))
    plt.xlabel('Effect size of ROI CT')
    plt.ylabel('Effect size of ROI GMV')
    plt.legend()
    plt.show()

# %%
gmvtoct

# %%

c
# %%
print(a)
print(b)

# %%
print(c)

# %%
print(d)

# %%
