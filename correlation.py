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
# Effect size with MMSE t-value
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import datasets
import os
from meta_analysis.main import csv_meta_analysis
from meta_analysis import utils
import nibabel as nib

csv_path = './data/mask/gmv_id.csv'
df = pd.read_csv(csv_path, index_col=1)
nii_path  = './data/mask/rBN_Atlas_246_1mm.nii'
nii = nib.load(nii_path)

centers = datasets.load_centers_all()
pairs = [(2,0)]
csv_prefixs = ['roi_gmv_removed/']

class Result(object):
    def __init__(self, r, p):
        self.r = r
        self.p = p

for pair in pairs:
    """
    ts = []
    for center in centers:
        mmse1, _ = center.get_MMSEs(label=pair[0])
        mmse2, *_ = center.get_MMSEs(label=pair[1])
        if mmse1 is not None and mmse2 is not None:
            t = ttest_ind(mmse1, mmse2).statistic
            ts.append(t)
    ts = ts
    """

    out_dir = './results/meta/{}_{}/correlation'.format(pair[0], pair[1])
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    mmse_path = './data/meta_csv/{}_{}/confound/MMSE.csv'.format(pair[0], pair[1])
    model1 = csv_meta_analysis(mmse_path, model_type='random')
    ts = model1.effect_sizes

    rs = []
    ps = []
    for csv_prefix in csv_prefixs:
        nii_array = np.asarray(nii.dataobj).astype(np.float64)
        p_array = nii_array

        csv_dir = './data/meta_csv/{}_{}/{}'.format(pair[0], pair[1], csv_prefix)
        csvs = os.listdir(csv_dir)
        models = {}
        for f in csvs:
            csv_path = os.path.join(csv_dir, f)
            model = csv_meta_analysis(csv_path, model_type='random')
            ess = model.effect_sizes
            r = pearsonr(ess, ts)[0]
            p = pearsonr(ess, ts)[1]
            rs.append(r)
            ps.append(p)
            models[f[:-4]] = Result(r, p)
        
        ll = [i for i in range(1, 247)]
        lowest = 1
        highest = 0
        for k, v in models.items():
            _id = df.loc[k]['ID']
            if v.p < 0.05:
                nii_array[nii_array==_id] = v.r
                p_array[p_array==_id] = v.p
                ll.remove(_id)
                print(k)
                if v.r < lowest:
                    lowest = v.r
                if v.r > highest:
                    highest = v.r
        for i in ll:
            nii_array[nii_array==i] = 0

        print(lowest)
        print(highest)

        path = os.path.join(out_dir, 'r.nii'.format(pair[0], pair[1]))
        p_path = os.path.join(out_dir, 'p.nii'.format(pair[0], pair[1]))
        utils.gen_nii(nii_array, nii, path)
        utils.gen_nii(nii_array, nii, p_path)

# %%
r_path = './results/meta/{}_{}/correlation/r.nii'.format(2, 0)
r_nii = nib.load(r_path)
mask_path = './results/meta/{}_{}/roi_gmv_removed/es_001_top30.nii'.format(2, 0)
mask_nii = nib.load(mask_path)

mask_array = np.asarray(mask_nii.dataobj).astype(np.float32)
mask_indics = mask_array == 0
r_array = np.asarray(r_nii.dataobj).astype(np.float32)
r_array[mask_indics] = 0
r_path = './results/meta/{}_{}/correlation/r_masked.nii'.format(2, 0)
utils.gen_nii(r_array, mask_nii, r_path)

# %%
ps = np.array(ps)
ps[ps<0.05]

# %%
# Correlation with PET
def min_max(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
gmv_id_path = './data/mask/gmv_id.csv'
roi_id_df = pd.read_csv(gmv_id_path, index_col=1)

pet_path = './data/PET/masked_mean'
files = os.listdir(pet_path)

es_path = './results/meta/2_0/roi_gmv_removed/TOP246.csv'
es_df = pd.read_csv(es_path, index_col=0)

ndf = pd.merge(es_df['es'], roi_id_df['ID'], on='name')
ndf.set_index('ID')
pet_path = './data/PET/masked_mean'
files = os.listdir(pet_path)
for f in files:
    path = os.path.join(pet_path, f)
    df = pd.read_csv(path, index_col=0)
    nndf = pd.merge(ndf, df, left_on='ID', right_on='id', left_index=True)

    #nndf = nndf[nndf['value'] < 30]

    x = nndf['es'].to_list()
    y = nndf['value'].to_list()
    nndf.to_csv('./data/PET/xy.csv')

    x = np.asarray(x)
    x = -x
    y = np.asarray(y)

    r = pearsonr(x, y)[0]
    p = pearsonr(x, y)[1]
    beta, bias = np.polyfit(x, y, deg=1)

    y_est = beta * x + bias
    y_err = y.std() * np.sqrt(1/len(y) +
                        (y - y.mean())**2 / np.sum((y - y.mean())**2))
    """
    plt.plot(x, y, 'o')
    plt.plot(x, y_est, 
            label='r={:.2f}, p={:.2e}'.format(r, p))
    plt.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    """
    sns.regplot(x=x,y=y, robust=True, label='r={:.2f}, p={:.2e}'.format(r, p))
    plt.xlabel('Effect size of ROI GMV')
    plt.ylabel(f)
    plt.legend()
    plt.show()
    
#%%
# Partial Correlation with PET
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from pingouin import partial_corr

roi_id_path = './data/mask/gmv_id.csv'
roi_id_df = pd.read_csv(roi_id_path, index_col=1)

es_path = './results/meta/2_0/roi_gmv_removed/TOP246.csv'
es_df = pd.read_csv(es_path, index_col=0)

ndf = pd.merge(es_df['es'], roi_id_df['ID'], on='name')
ndf.set_index('ID')

cov_path = './data/mask/tpm.csv'
cov_df = pd.read_csv(cov_path, index_col=0)

pet_path = './data/PET/masked_mean'
files = os.listdir(pet_path)
for f in files:
    path = os.path.join(pet_path, f)
    df = pd.read_csv(path, index_col=0)
    nndf = pd.merge(ndf, df, left_on='ID', right_on='id', left_index=True)
    nndf = pd.merge(nndf, cov_df, left_on='id', right_on='id', left_index=True)
    ans = partial_corr(nndf, x='es', y='value', covar='tpm')
    print(f,ans.iloc[0]['r'])
# %%
ans