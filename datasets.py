"""
Module Docstring
TODO: add license
"""

import os

import center
import mask

def load_masks(mask_dir='./data/DMN_region'):
    """load masks from disk
    Args:
        mask_dir: string, dir of masks

    Returns:
        Masks instance, details in mask.py
    """
    return mask.Masks(mask_dir)

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

def load_centers_all(filenames='origin.csv'):
    centers_mcad = load_centers_mcad(filenames)
    centers_edsd = load_centers_edsd(filenames)
    centers_adni = load_centers_adni(filenames)
    return centers_mcad + centers_edsd + centers_adni
