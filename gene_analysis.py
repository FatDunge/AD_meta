"""
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
"""
# %%
# PLSR
from scipy import stats
import pandas as pd
from pyls import pls_regression
def plsr(roi_models, n_components=2, 
         n_perm=5000, n_boot=5000,
         gene_path='./data/gene/expression.csv',
         out_path='./data/gene/expression_plsr.csv'):
    roi_es_dict = {}
    for k, v in roi_models.items():
        roi_es_dict[int(k)] = v.total_effect_size

    roi_df = pd.DataFrame.from_dict(roi_es_dict, orient='index', columns=['es'])

    gene_df = pd.read_csv(gene_path, index_col=0)
    
    es_filtered = roi_df[roi_df.index.isin(gene_df.index)]
    gene_filtered = gene_df[gene_df.index.isin(es_filtered.index)]
    x = gene_filtered.values
    y = es_filtered.values
    
    x = stats.zscore(x)
    y = stats.zscore(y)

    plsr = pls_regression(x, y, n_components=n_components, n_perm=n_perm,
                          n_boot=n_boot)

    pls1 = plsr.x_weights.T[0]
    pls2 = plsr.x_weights.T[1]
    gene_name = list(gene_df.columns)
    d = {'gene_name':gene_name, 'pls1':pls1, 'pls2':pls2}
    df = pd.DataFrame(d)
    df.set_index('gene_name')
    df.to_csv(out_path)
    return plsr

#%%
import seaborn as sns
def cfg_distribution(x):
    sns.distplot(x)