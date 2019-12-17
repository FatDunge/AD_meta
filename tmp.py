#%%
with open('./tmp/filenames.txt') as f:
    t = f.read().splitlines()
v = [0] * 1779
t = dict(zip(t, v))
# %%
import os
import xml.etree.ElementTree as ET
path = r'E:\kxp\data\adni_reg\report'
proed = os.listdir(path)
for filename in proed:
    if 'txt' in filename:
        txt = os.path.join(path, filename)
        with open(txt) as f:
            st = f.read()
            if 'error' in st or '无法' in st:
                print(filename)
# %%