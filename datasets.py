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

def load_centers_mcad(use_nii=False, use_csv=False,
                      use_personal_info=False, use_xml=False,
                      nii_prefix='mri/{}.nii',
                      personal_info_prefix='personal_info/{}.csv',
                      csv_prefix='csv/{}.csv'):
    """get list of MCAD's Center
    Args:
        use_nii: bool, whether to load nii
        use_gm: bool, whether to load segmented gm nii
        use_wm: bool, whether to load segmented gm nii
        use_csv: bool, whether to load csv
        use_xml: bool, whether to load xml

    Returns:
        list of Center
    """
    centers = []
    for i in range(1, 9):
        center_path = './data/AD/MCAD/AD_S0{0}/AD_S0{0}_MPR'.format(i)
        _center = center.CenterCAT(center_path, 'filenames.txt',
                                   use_nii=use_nii, use_csv=use_csv, 
                                   use_personal_info=use_personal_info,
                                   use_xml=use_xml,
                                   nii_prefix=nii_prefix,
                                   personal_info_prefix=personal_info_prefix,
                                   csv_prefix=csv_prefix)
        centers.append(_center)
    return centers

def load_centers_edsd(data_path='./data/AD/EDSD/EDSD_T1',
                      use_nii=False, use_csv=False,
                      use_personal_info=False, use_xml=False,
                      nii_prefix='mri/{}.nii',
                      personal_info_prefix='personal_info/{}.csv',
                      csv_prefix='csv/{}.csv'):
    """get list of EDSD's Center
    Args:
        use_nii: bool, whether to load nii
        use_csv: bool, whether to load csv
        use_xml: bool, whether to load xml

    Returns:
        list of Center
    """
    centers = []
    center_names = os.listdir(data_path)

    for center_name in center_names:
        center_path = os.path.join(data_path, center_name)
        _center = center.CenterCAT(center_path, 'filenames.txt',
                                   labels=['HC', 'MC', 'AD'],
                                   use_nii=use_nii, use_csv=use_csv,
                                   use_personal_info=use_personal_info,
                                   use_xml=use_xml,
                                   nii_prefix=nii_prefix,
                                   personal_info_prefix=personal_info_prefix,
                                   csv_prefix=csv_prefix)
        centers.append(_center)
    return centers

if __name__ == "__main__":
    pass
