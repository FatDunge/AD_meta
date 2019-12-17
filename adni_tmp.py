#%%
import pandas as pd
import re
csv_path = './data/center_info/ADNI/a.csv'

df = pd.read_csv(csv_path)
subjects = []

for index, row in df.iterrows():
    filename = row['filename']
    pattern = re.compile(r'\d{3}_S_\d{4}')   # 查找数字
    result = pattern.findall(filename)
    subjects.append(result[0])
print(len(subjects))
#%%
from collections import Counter
centers = []
for subject in subjects:
    center = subject[:3]
    centers.append(center)
centers = Counter(centers)
print(centers)

a = []
b = []
for k, v in centers.items():
    if v > 30:
        a.append(k)
    elif v > 20: 
        b.append(k)
print(len(a))
print(len(b))


# %%
