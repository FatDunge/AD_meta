#%%
# load roi effect sizes
import pandas as pd
from scipy.stats import pearsonr
from numpy.polynomial.polynomial import polyfit
import numpy as np
import csv
import matplotlib.pyplot as plt
class Result(object):
    def __init__(self, gene_name, effect_sizes, gene_expression):
        self.gene_name = gene_name
        self.effect_sizes = effect_sizes
        self.gene_expression = gene_expression
        rp = pearsonr(effect_sizes, gene_expression)
        self.r = rp[0]
        self.p = rp[1]

def sort_results(results, orderby='r',descend=True):
    if orderby == 'r':
        list1 = [result.r for result in results]
    else:
        list1 = [result.r for result in results]
    list1, results = (list(t) for t in zip(*sorted(zip(list1, results), reverse=descend)))
    return results

pair = (2, 0)
es_prefixs = ['roi_gmv_removed/TOP246.csv', 'roi_ct_removed/TOP210.csv']
roi_id_paths = ['./data/mask/gmv_id.csv', './data/mask/cortical_id.csv']

gene_path = './data/gene/expression_cfg5.csv'
gene_df = pd.read_csv(gene_path, index_col=0)

norm = True
def min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

out_paths = ['./results/gene/correlation_gmv_gene.csv', './results/gene/correlation_ct_gene.csv']

for es_prefix, roi_id_path, out_path in zip(es_prefixs, roi_id_paths, out_paths):
    results = []
    es_path = './results/meta/{}_{}/{}'.format(pair[0], pair[1], es_prefix)
    es_df = pd.read_csv(es_path, index_col=0)
    roi_id_df = pd.read_csv(roi_id_path, index_col='name')
    ndf = pd.merge(es_df['es'], roi_id_df['ID'], on='name')
    with open(out_path, 'w', newline='') as file:
        fieldnames = ['gene_name', 'r','p','sign']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, value in gene_df.iteritems():
            ss = pd.merge(ndf, value, left_on='ID', right_on='label', left_index=True)
            ss = ss.drop('ID',axis=1)
            gene_name = ss.columns[1]
            a = ss.values.T[0]
            b = ss.values.T[1]
            if norm:
                a = min_max(a)
                b = min_max(b)
            """
            if gene_name == 'PSN':
                plt.plot(a, b, '.')
                r = pearsonr(a, b)[0]
                p = pearsonr(a, b)[1]
                bias, m = polyfit(a, b, 1)
                plt.plot(a, bias + m * a, 
                        label='r={:.2f}, p={:.2e}'.format(r, p))
                plt.xlabel('Effect size of ROI GMV')
                plt.ylabel('Gene Expression of ROI')
                plt.legend()
                plt.show()
            """
            r = pearsonr(a, b)[0]
            p = pearsonr(a, b)[1]

            if p < 0.001 / len(gene_df.columns):
                sign = '***'
            elif p < 0.01 / len(gene_df.columns):
                sign = '**'
            elif p < 0.05 / len(gene_df.columns):
                sign = '*'
            else:
                sign = 'NS'
            
            writer.writerow({'gene_name': gene_name,
                            'r': r,
                            'p': p,
                            'sign': sign})
        
            results.append(Result(gene_name, a, b))
    print(len(results))
# %%
import pandas as pd
gene_path = './data/gene/expression.csv'
save_path = './data/gene/expression_{}.csv'.format('cfg5')
gene_df = pd.read_csv(gene_path, index_col=0)

gene_list = ['AP2A2','APOE','ARHGDIB','CAV1','CDH4','CDH13',
             'EPS15','FLT1','NRG1','MAPT','MME','MAPK10','RAPSN',
             'CUL1','NR1H3','PAK4','CAMKK2']

total_list = list(gene_df.columns)

def list_contains(list1, list2): 
    for m in list1:
        check = False
        for n in list2: 
            if m == n:
                check = True
        if not check:
            list1.remove(m)
    return list1

geng_list = list_contains(gene_list, total_list)

df1 = gene_df[gene_list]
df1.to_csv(save_path)

# %%
# PLSR
import pandas as pd
gene_path = './data/gene/expression.csv'
gene_df = pd.read_csv(gene_path, index_col=0)
t = gene_df.values
pair = (2, 0)
es_prefix = 'roi_gmv_removed/TOP246.csv'
es_path = './results/meta/{}_{}/{}'.format(pair[0], pair[1], es_prefix)
es_df = pd.read_csv(es_path, index_col=0)

roi_id_path = './data/mask/gmv_id.csv'
roi_id_df = pd.read_csv(roi_id_path, index_col='name')

ndf = pd.merge(es_df['es'], roi_id_df['ID'], on='name').set_index('ID')
es_filtered = ndf[ndf.index.isin(gene_df.index)]
c = es_filtered.values
# %%
from sklearn.cross_decomposition import PLSRegression
pls2 = PLSRegression(n_components=2)
pls2.fit(t, c)

# %%
pls2.x_weights_[:,0].shape
#%%
pls1 = pls2.x_weights_[:,0]
gene_name = list(gene_df.columns)
d = {'gene_name':gene_name,'pls1':pls1}
df = pd.DataFrame(d, index)
df.to_csv('./data/gene/expression_pls1.csv')
# %%
from sklearn.model_selection import permutation_test_score
score, permutation_scores, pvalue = permutation_test_score(
    pls2, t, c, scoring="accuracy", n_permutations=100, n_jobs=1)


# %%
len()

# %%
import nilearn as nil
import nibabel as nib
import os

src_dir = r'H:\workspace\JuSpace\Juspace_v1\PETatlas'
to_dir = r'./data/PET'
temp = r'./data/mask/ch2bet.nii'
temp_nii = nib.load(temp)
fs = os.listdir(src_dir)
for f in fs:
    src_path = os.path.join(src_dir, f)
    to_path = os.path.join(to_dir, f)
    src_nii = nib.load(src_path)
    to_nii = nil.image.resample_to_img(src_nii, temp_nii)
    nib.save(to_nii, to_path)

# %%
