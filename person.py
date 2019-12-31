"""
TODO: add license
"""
import os
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nibabel as nib
from nibabel import processing
from nibabel import nifti1

def get_nii_data(nii):
    return np.asarray(nii.dataobj)

class FilenameError(Exception):
    """Exception in case filename don't match
    """
    def __init__(self, filename):
        super(FilenameError, self).__init__()
        self.filename = filename

    def __str__(self):
        return repr(self.filename)

class FileNotLoadError(Exception):
    """docstring for FileNotLoadError"""
    def __init__(self, filename):
        super(FileNotLoadError, self).__init__()
        self.filename = filename

    def __str__(self):
        return repr(self.filename)

class Person(object):
    """docstring for Person
    different software preprocessed data should implement this

    Attributes:
        file_dir: string, center's dir
        filename: string, person's no
        nii_prefix: string, full name of nii file lack of person's no
        csv_prefix: string, full name of region volume csv file lack of person's no
        presonal_info_prefix: string, full name of personal info csv file lack of person's no
        xml_prefix: string, full name of xml file lack of person's no

    Methods to be implemented:
        create_csv
        load_nii
        load_csv
        load_report
        get_tiv
        get_total_cgw_volume
    """
    def __init__(self, file_dir, filename,
                 labels=['NC', 'MC', 'AD'],
                 nii_prefix='mri/{}.nii',
                 csv_prefix='csv/{}.csv',
                 personal_info_prefix='personal_info/{}.csv',
                 xml_prefix='report/cat_{}.xml'):
        super(Person, self).__init__()
        self.file_dir = file_dir
        self.filename = filename
        self.labels = labels
        self.nii_prefix = nii_prefix
        self.csv_prefix = csv_prefix
        self.personal_info_prefix = personal_info_prefix
        self.xml_prefix = xml_prefix

        self.label = None
        self.nii = None
        self.grey_matter = None
        self.white_matter = None
        self.personal_info = None
        self.dataframe = None
        self.report = None

    def create_csv(self, masks_dict):
        """inherited class should override this method
        """
        raise NotImplementedError('create_csv')

    def load_nii(self, prefix):
        """inherited class should override this method
        """
        raise NotImplementedError('load_nii')

    def load_csv(self):
        """inherited class should override this method
        """
        raise NotImplementedError('load_csv')

    def load_report(self):
        """inherited class should override this method
        """
        raise NotImplementedError('load_report')

    def get_tiv(self):
        """inherited class should override this method
        """
        raise NotImplementedError('get_tiv')

    def get_total_cgw_volume(self):
        """inherited class should override this method
        """
        raise NotImplementedError('get_total_cgw_volume')

def get_volume(mask_data, data):
    """apply a mask to file then calculate volume

        Args:
            mask_data: ndarray
            data: ndarray, shape should be the same with $mask_data$

        Returns:
            volume of data
    """
    assert mask_data.shape == data.shape
    masked_data = np.multiply(mask_data, data)
    return np.sum(masked_data)

class PersonCAT(Person):
    """docstring for PersonCAT
    create instance for CAT12 preprocessed data

    Attributes:
        use_nii: bool, whether to load nii
        use_gm: bool, whether to load segmented gm nii
        use_wm: bool, whether to load segmented gm nii
        use_csv: bool, whether to load csv
        use_presonal_info: bool, whether to load personal info
        use_xml: bool, whether to load xml
        nii: niimage, person's niimage
        gm: niimage, person's segmented gm niimage
        wm: niimage, person's segmented wm niimage
        label: int, 0 for NC, 1 for MCI, 2 for AD
        dataframe: dataframe, region volume load from csv
        report: ElementTree, person's CAT analysis report
    """

    def __init__(self, file_dir, filename,
                 labels=['NC', 'MC', 'AD'],
                 use_nii=False, use_csv=True, 
                 use_personal_info=False, use_xml=False,
                 nii_prefix='mri/wm{}.nii',
                 csv_prefix='csv/{}.csv',
                 personal_info_prefix='personal_info/{}.csv',
                 xml_prefix='report/cat_{}.xml'):
        super(PersonCAT, self).__init__(file_dir, filename, labels,
                                        nii_prefix, csv_prefix,
                                        personal_info_prefix, xml_prefix)

        for label in labels:
            if label in self.filename:
                self.label = labels.index(label)
        #filename should contain one of these flags
        if self.label is None:
            raise FilenameError(filename)

        if use_nii:
            self.nii = self.load_nii(nii_prefix)
        if use_csv:
            self.dataframe = self.load_csv(csv_prefix)
        if use_personal_info:
            self.personal_info = self.load_csv(personal_info_prefix, index_col=None)
        if use_xml:
            self.report = self.load_report()

    def load_nii(self, prefix):
        nii_path = os.path.join(self.file_dir,
                                prefix.format(self.filename))
        return nib.load(nii_path)

    def load_csv(self, prefix, index_col=0):
        csv_path = os.path.join(self.file_dir,
                                prefix.format(self.filename))
        return pd.read_csv(csv_path, index_col=index_col)

    def load_report(self):
        xml_path = os.path.join(self.file_dir,
                                self.xml_prefix.format(self.filename))
        return ET.parse(xml_path)

    def create_csv(self, mask):
        """use masks to calculate region's volume
        save GMV in csv file

        Args:
            
        """
        _min, _max = mask.get_min_max_label()

        csv_path = os.path.join(self.file_dir,
                                self.csv_prefix.format(self.filename))
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['ID', 'GMV']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(_min, _max+1):
                if i == 0:
                    pass
                else:
                    gmv = mask.get_masked_volume(self.nii, i)
                    writer.writerow({'ID': i, 'GMV': gmv})
    
    def create_other_csv(self, values, csv_prefix='csv_removed/{}.csv'):
        csv_path = os.path.join(self.file_dir,
                                csv_prefix.format(self.filename))
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['ID', 'GMV']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            i = 1
            for value in values:
                writer.writerow({'ID': i, 'GMV': value})
                i += 1

    def get_label_binary(self):
        if self.label == 0:
            return [1, 0, 0]
        elif self.label == 1:
            return [0, 1, 0]
        else:
            return [0, 0, 1]
        return None

    def get_tiv(self):
        if self.report:
            root = self.report.getroot()
            vol_tiv = root.findall('./subjectmeasures/vol_TIV')[0].text
        else:
            raise FileNotLoadError(self.xml_prefix.format(self.filename))
        return float(vol_tiv)

    def get_total_cgw_volume(self):
        """get brain total CSF, GM, WM volume

        Returns:
            list, [CSF, GMV, WMV]
        """
        if self.report:
            root = self.report.getroot()
            cgw = root.findall('./subjectmeasures/vol_abs_CGW')
            tmp = [float(i) for i in cgw[0].text.replace('[', '').replace(']', '').split()]
        else:
            raise FileNotLoadError(self.xml_prefix.format(self.filename))
        return tmp[0:3]

    def get_presonal_info_values(self):
        """get personal info file value
        
        Returns: list, first line of personal info file, [age, male, female, MMSE]
        """
        if self.personal_info is not None:
            values = self.personal_info.values[0]
        else:
            raise FileNotLoadError('personal_info')
        return values

    def get_region_volume(self, region, tissue_type='GMV', use_tiv=False):
        """get DMN region's volume

        Args:
            region: string, region's filename
            tissue_type: string, indicate tissue_type to get from $dataframe$
            use_tiv: bool, whether to divide tiv while calculating region volume

        Returns:
            float, region's volume
        """
        if self.dataframe is not None:
            if use_tiv:
                volume = self.dataframe[tissue_type][region]/self.get_tiv()
            else:
                volume = self.dataframe[tissue_type][region]
        else:
            raise FileNotLoadError('csv')
        return volume

    def save_smoothed_image(self, nii, fwhm=4, mode='nearest',
                            prefix='mri_smoothed/{}.nii'):
        """smooth image using nibabel and save to disk

        Args:
            nii: nii file to smooth
            fwhm: int or length 3 sequence, full-width at half-maximum(FWHM), size of guassian kernel
            mode: string, Points outside the boundaries of the input are filled according to the given mode
                  (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). 
            prefix: string, where to save smoothed image
        """
        smoothed_image = processing.smooth_image(nii, fwhm, mode=mode)
        filename = prefix.format(self.filename)
        filepath = os.path.join(self.file_dir, filename)
        nifti1.save(smoothed_image, filepath)

    def save_image(self, nii, prefix):
        filename = prefix.format(self.filename)
        filepath = os.path.join(self.file_dir, filename)
        nifti1.save(nii, filepath)
