#%%
import pandas as pd
import os
import csv

df = pd.read_excel('./data/center_info/EDSD/edsd_a.xlsx', sheet_name='1', index_col=2)

for index, row in df.iterrows():
    data_path = './data/AD/EDSD/EDSD_T1'
    center_path = os.path.join(data_path, index[:3])
    file_path = os.path.join(center_path, 'personal_info', '{}.csv'.format(index))
    if row['gender'] == 'male':
        male = 1
        female = 0
    else:
        male = 0
        female = 1
    
    
    with open(file_path, 'w', newline='') as file:
        fieldnames = ['age', 'male', 'female', 'MMSE']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({'age': row['age']/100,
                         'male': male, 'female':female,
                         'MMSE':row['MMSE']})


#%%
import pandas as pd
import os
import csv
i = 3
sex = '性别'
age = '年龄'
MMSE = 'MMSE'
df = pd.read_excel('./data/center_info/MCAD/mcad.xlsx', sheet_name='AD_S0{}'.format(i),
                   index_col=0)
for index, row in df.iterrows():
    file_path = './data/AD/MCAD/AD_S0{0}/AD_S0{0}_MPR/personal_info/{1}.csv'.format(i, index)
    if row[sex] == 1:
        male = 1
        female = 0
    else:
        male = 0
        female = 1
    
    with open(file_path, 'w', newline='') as file:
        fieldnames = ['age', 'male', 'female', 'MMSE']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({'age': row[age]/100,
                         'male': male, 'female':female,
                         'MMSE': row[MMSE]})

#%%
