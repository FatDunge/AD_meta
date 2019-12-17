#%%
import os
import pandas as pd
path = r'C:\Users\dunge\Downloads\20191004\ADNI'
csv_path = r'C:/Users/dunge/Downloads/filenames.csv'
df = pd.read_csv(csv_path)
count = 0
for dirpath,dirnames,filenames in os.walk(path):
    if filenames:
        for filename in filenames:
            for index, row in df.iterrows():
                filename_sub = filename[-40:]
                filename_sub2 = row['filename'][-40:]
                if filename_sub == filename_sub2:
                    count += 1
                    filepath = os.path.join(dirpath, filename)
                    re_path = os.path.join(dirpath, row['filename'])
                    os.rename(filepath, re_path)
print(count)
#%%
import os
import pandas as pd
path = r'C:\Users\dunge\Downloads\20191004\ADNI'
csv_path = r'C:/Users/dunge/Downloads/filenames.csv'
df = pd.read_csv(csv_path)
count = 0
for dirpath,dirnames,filenames in os.walk(path):
    if filenames:
        for filename in filenames:
            for index, row in df.iterrows():
                filename_sub = filename[-40:]
                filename_sub2 = row['filename'][-40:]
                if filename_sub == filename_sub2:
                    count += 1
                    from_path = os.path.join(dirpath, filename)
                    to_path = row['filename']
                    os.system('scp {} xpkang@172.18.31.107:{}'.format(from_path, to_path))

#%%
