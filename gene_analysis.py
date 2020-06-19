#%%
# load roi effect sizes
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import csv
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

gene_path = './data/gene/expression.csv'
gene_df = pd.read_csv(gene_path, index_col=0)

i = 0
results = []

out_paths = ['./results/gene/correlation_gmv.csv', './results/gene/correlation_ct.csv']

for es_prefix, roi_id_path, out_path in zip(es_prefixs, roi_id_paths, out_paths):
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

            r = pearsonr(a, b)[0]
            p = pearsonr(a, b)[1]

            if p < 0.001 / 15633:
                sign = '***'
            elif p < 0.01 / 15633:
                sign = '**'
            elif p < 0.05 / 15633:
                sign = '*'
            else:
                sign = 'NS'
            
            writer.writerow({'gene_name': gene_name,
                            'r': r,
                            'p': p,
                            'sign': sign})
        
            results.append(Result(gene_name, a, b))

# %%
