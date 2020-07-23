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
    def __init__(self, filename, data):
        self.filename = filename
        self.data = data
        self.shape = data.shape
        labels = np.unique(data)
        # Remove 0 as it usually don't act as label
        self.labels = labels[labels!=0]
        self.indices = np.transpose(np.nonzero(self.data))

    def get_label_bool(self, label):
        """get bool array by label
        Args:
            label: int, label in self.labels
        Returns:
            ndarray: bool array of label
        """
        assert label in self.labels
        # Bool element will be treated as 1 and 0 for further calculation
        return self.data==label

    def get_masked_data(self, array, label):
        """return masked array
        Args:
            array: ndarray
            label: mask label
        Return:
            masked array
        """
        bool_array = self.get_label_bool(label)
        return np.multiply(array, bool_array)

    def get_masked_volume(self, array, label):
        return np.sum(self.get_masked_data(array, label))

    def get_masked_mean(self, array, label):
        summed = np.sum(self.get_masked_data(array, label))
        n = np.count_nonzero(self.get_masked_data(array, label))
        return summed/n

    def get_all_masked_volume(self, array):
        volumes = {}
        labels = self.labels
        for i in labels:
            volumes[i] = self.get_masked_volume(array, i)
        return volumes

    def get_all_masked_mean(self, array):
        means = {}
        labels = self.labels
        for i in labels:
            means[i] = self.get_masked_mean(array, i)
        return means

class NiiMask(Mask):
    def __init__(self, file_dir, filename):
        self.file_dir = file_dir
        filepath = os.path.join(file_dir, filename)
        self.nii = nib.load(filepath)
        super().__init__(filename, np.asarray(self.nii.dataobj))

    def get_all_masked_volume(self, nii):
        array = np.asarray(nii.dataobj)
        return super().get_all_masked_volume(array)

    def get_all_masked_mean(self, nii):
        array = np.asarray(nii.dataobj)
        return super().get_all_masked_mean(array)
