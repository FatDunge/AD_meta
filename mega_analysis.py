#%%
import nibabel as nib
from nibabel.analyze import AnalyzeImage
from nilearn import plotting
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib
from nibabel import nifti1
from scipy.stats import norm
import nilearn as nil

import os
import csv

import datasets
import mask
import meta
#%%
SETTINGS = {"datatype":"CONT",
            "models":"Random",
            "algorithm":"IV-Cnd",
            "effect":"SMD"}

def create_es_p_image(studies, n, nii, correction=False):
    shape = (181, 217, 181)
    x, y , z = shape
    count = x * y * z
    es_image = np.zeros(shape=(count))
    lci_image = np.zeros(shape=(count))
    uci_image = np.zeros(shape=(count))
    se_image = np.zeros(shape=(count))
    i2_image = np.zeros(shape=(count))
    z_image = np.zeros(shape=(count))
    p_image = np.zeros(shape=(count))

    if not correction:
        n = 1

    for k, v in studies.items():
        result = meta.meta(v, SETTINGS, plot_forest=False)

        z = float(result[0][10])
        p = norm.sf(z) * 2

        if p < 0.05/n:
            es_image[k] = result[0][1]
            lci_image[k] = result[0][3]
            uci_image[k] = result[0][4]
            se_image[k] = result[0][6]
            i2_image[k] = result[0][9]
            z_image[k] = z
            p_image[k] = -np.log10(p)

    def create_nii(image, shape, affine, header):
        image = np.reshape(image, shape)
        return nib.Nifti1Image(image, affine, header)

    affine = nii.affine
    header = nii.header
    header.set_data_dtype(np.float32)

    es_nii = create_nii(es_image, shape, affine, header)
    lci_nii = create_nii(lci_image, shape, affine, header)
    uci_nii = create_nii(uci_image, shape, affine, header)
    se_nii = create_nii(se_image, shape, affine, header)
    i2_nii = create_nii(i2_image, shape, affine, header)
    z_nii = create_nii(z_image, shape, affine, header)
    p_nii = create_nii(p_image, shape, affine, header)
    nii = nil.image.concat_imgs([es_nii, lci_nii, uci_nii, se_nii, i2_nii, z_nii, p_nii])
    return nii

#%%
_mask = mask.Mask('./data/mask', 'grey_matter_smoothed_005.nii')
d = _mask.get_mask_data().flatten()
n = len(d[d>0])
nii_prefix = 'mri_smoothed/{}.nii'
filenames = 'HYDRA.csv'
centers_mcad = datasets.load_centers_mcad(use_nii=True, use_csv=False,
                                          filenames=filenames, nii_prefix=nii_prefix)
centers_edsd = datasets.load_centers_edsd(use_nii=True, use_csv=False,
                                          filenames=filenames, nii_prefix=nii_prefix)
centers_adni = datasets.load_centers_adni(use_nii=True, use_csv=False,
                                          filenames=filenames, nii_prefix=nii_prefix)
#%%
centers = ['ALL']
corrections = [True]
labels = ['NC', 'MC', 'AD-1', 'AD-2']
pairs = [(2,0), (3, 0), (3, 2)]
#%%
for center in centers:
    if center == 'EDSD':
        centers_list = centers_edsd
    elif center == 'MCAD':
        centers_list = centers_mcad
    elif center == 'ADNI':
        centers_list = centers_adni
    elif center == 'ALL':
        centers_list = centers_edsd + centers_mcad + centers_adni

    for _center in centers_list:
        if len(_center.persons) <= 20:
            centers_list.remove(_center)

    for pair in pairs:
        label_eg = pair[0]
        label_cg = pair[1]
        studies = meta.gen_voxel_studies(centers_list, _mask, label_eg, label_cg)
        for correction in corrections:
            nii = create_es_p_image(studies, n,
                                    centers_list[0].persons[0].nii, correction=correction)
            niis = {'4D': nii}

            if correction:
                tmp = 'with'
            else:
                tmp = 'without'
            
            for key, nii in niis.items():
                path = './report/{}/voxelwise_{}_{}{}_{}_correction.'.format(center, key,
                                                                             labels[label_eg],
                                                                             labels[label_cg], tmp)
                nii_path = path + 'nii'
                nifti1.save(nii, nii_path)

#%%
