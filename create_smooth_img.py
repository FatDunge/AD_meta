# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%

import datasets
centers = datasets.load_centers_mcad(use_csv=False, use_xml=False,
                                     use_personal_info=True)
#%%
for person in centers[2].persons:
    person.save_smoothed_image(person.grey_matter)


#%%
"""
import datasets
centers = datasets.load_centers_edsd(use_gm=True, use_csv=False, use_xml=False)

for center in centers:
    for person in center.persons:
        person.save_smoothed_image(person.grey_matter)
"""


#%%
