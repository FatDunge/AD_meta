"""
Module Docstring
TODO: add license
"""

import os

import center
import mask

def load_centers(center_dir, filenames):
    centers = []
    center_names = os.listdir(center_dir)

    for center_name in center_names:
        center_path = os.path.join(center_dir, center_name)
        _center = center.Center(center_path, filenames)
        centers.append(_center)
    return centers

def load_centers_mcad(filenames='origin.csv'):
    center_dir = './data/AD/MCAD'
    centers = load_centers(center_dir, filenames)
    return centers

def load_centers_edsd(filenames='origin.csv'):
    center_dir = './data/AD/EDSD'
    centers = load_centers(center_dir, filenames)
    return centers

def load_centers_adni(filenames='origin.csv'):
    center_dir = './data/AD/ADNI'
    centers = load_centers(center_dir, filenames)
    return centers

def load_centers_adni_merge(filenames='origin.csv'):
    center_dir = './data/AD/ADNI_merge'
    centers = load_centers(center_dir, filenames)
    return centers

def load_centers_all(filenames='origin.csv', adni_merge=True):
    centers_mcad = load_centers_mcad(filenames)
    centers_edsd = load_centers_edsd(filenames)
    if adni_merge:
        centers_adni = load_centers_adni_merge(filenames)
    else:
        centers_adni = load_centers_adni(filenames)
    return centers_mcad + centers_edsd + centers_adni
