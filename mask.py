"""
TODO:add license
"""

import os
import nibabel as nib
import numpy as np

def nomalize(data):
    """nomalize data using global min,max
    """
    _min, _max = data.min(), data.max()
    return (data-_min)/(_max-_min)

class Mask(object):
    """docstring for Mask

    Attributes:
        file_dir: file dir that this mask are in
        filename: mask's filename
        nii: mask's niimage(Nifti1Image)
    """
    def __init__(self, file_dir, filename):
        super(Mask, self).__init__()
        self.file_dir = file_dir
        self.filename = filename
        mask_path = os.path.join(file_dir, filename)
        self.nii = nib.load(mask_path)
        self.data = np.asarray(self.nii.dataobj)

    def get_mask_data(self, label=None):
        """get this mask's array data

        Returns:
            ndarray
        """
        if label is None:
            mask_data = nomalize(self.data)
        else:
            mask_data = self.data==label
        return mask_data

    def get_min_max_label(self):
        _min = np.min(self.data)
        _max = np.max(self.data)
        return _min, _max

    def get_masked_image_data(self, image, label=None, mode='array'):
        """
            mode: string, 
                'array': numpy array, same shape with mask data
                'nii': nibabel nifti1 instance
        """
        mask_data = self.get_mask_data(label)
        data = np.asarray(image.dataobj)
        return np.multiply(mask_data, data)

    def get_masked_volume(self, image, label):
        return np.sum(self.get_masked_image_data(image, label))


class Masks(object):
    """class to manage all Mask instance

    Attributes:
        file_dir: file dir contains all mask's file
        masks_dict: dict of all mask
    """
    def __init__(self, file_dir):
        super(Masks, self).__init__()
        self.file_dir = file_dir
        self.masks_dict = self.load_masks()

    def load_masks(self):
        """load masks as a dict contains Mask instance
        file_dir must not have other file

        Returns:
            dict, {filename: Mask, ...}
        """
        mask_filenames = os.listdir(self.file_dir)
        masks_dict = {}
        for filename in mask_filenames:
            mask = Mask(self.file_dir, filename)
            masks_dict[filename] = mask
        return masks_dict

    def get_masks_name(self):
        """get all mask's filename, aka region's name.

        Returns:
            list of string, all keys
        """
        return self.masks_dict.keys()

    def get_masks_data(self):
        """get all mask's array data

        Returns:
            dict, {mask_name: ndarray}
        """
        masks_data_dict = {}
        for k, value in self.masks_dict.items():
            masks_data_dict[k] = value.get_mask_data()
        return masks_data_dict
