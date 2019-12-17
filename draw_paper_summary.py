#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def bar_plot(x, title=''):
    c = Counter(x)
    fig, ax = plt.subplots()
    ax.bar(c.keys(), c.values())
    ax.set_title(title)
    plt.show()

def find_str_before(string, before='('):
    return string[:string.find(before)]

def pie_plot(x, title=''):
    c = Counter(x)
    fig, ax = plt.subplots()
    ax.pie(c.values(), labels=c.keys())
    ax.set_title(title)
    plt.show()

#%%

file_path = './report/PaperSummary.xlsx'
sheet_names = ['AD', 'MCI']
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name)
    last_index = None
    years = []
    voxels = []
    Ps = []
    n_nc = []
    n_ad = [] 
    for i in df.index:
        year = df.iloc[i]['year']
        study = df.iloc[i]['study']
        voxel = df.iloc[i]['voxel']
        p = str(df.iloc[i]['P'])
        cor = df.iloc[i]['correction']
        if not pd.isnull(year):
            years.append(int(year))
            voxels.append(find_str_before(voxel, before='x'))
            if '/' in p:
                Ps.append('Multi')
            else:
                Ps.append(p+'/'+cor)
        if not pd.isnull(study):
            last_index = i
            #nc = int(find_str_before(df.iloc[i]['Number(Female)']))
            #ad = int(find_str_before(df.iloc[i]['Unnamed: 4']))
            #n_nc.append(nc)
            #n_ad.append(ad)
        elif pd.isnull(study) and last_index is not None:
            """
            nc = int(find_str_before(df.iloc[i]['Number(Female)'])))
            ad = int(find_str_before(df.iloc[i]['Unnamed: 4']))
            n_nc.append(n_nc.pop()+nc)
            n_ad.append(n_ad.pop()+ad)
            """
            pass

    bar_plot(years, title='years')
    bar_plot(voxels, title='voxel size')
    pie_plot(Ps, title='P')

#%%
