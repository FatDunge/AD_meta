"""
TODO: add license
"""
import ast
import csv
import os
import re
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_morph_data
from nibabel import nifti1

def load_array(path, dtype=np.float32):
    nii = nib.load(path)
    array = np.asarray(nii.dataobj, dtype=dtype)
    array = np.nan_to_num(array)
    return array

class Person(object):
    def __init__(self, filename, label):
        super(Person, self).__init__()
        self.filename = filename
        self.label = label

class Center(object):
    """docstring for Center

    Attributes:
        file_dir: string, center's dir
        filenames: string, filepath to a csv file contains all person's no and their label
        use_nii: bool, whether to load nii
        use_csv: bool, whether to load csv
        use_xml: bool, whether to load xml
        persons: list of Person
    """

    def __init__(self, file_dir, filenames='origin.csv'):
        name = file_dir[file_dir.rfind('/')+1:]
        self.file_dir = file_dir
        self.name = name
        self.filenames = filenames
        self.persons = self.load_persons()

    def load_persons(self):
        """get list of Person

        Returns:
            list of Person
        """
        persons = []
        csv_path = os.path.join(self.file_dir, self.filenames)
        #get person's filename in txt file
        df = pd.read_csv(csv_path, index_col=0)
        for index, value in df.iterrows():
            filename = index
            label = value['label']
            _person = Person(filename, label)
            persons.append(_person)
        return persons

    def save_labels(self, filename):
        path = os.path.join(self.file_dir, filename)
        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['filename', 'label'])
            writer.writeheader()
            for person in self.persons:
                writer.writerow({'filename':person.filename,
                                 'label':person.label})

    def get_by_label(self, label):
        """get list of Person in specified label

        Returns:
            list of Person
        """
        tmp = []
        if self.persons:
            for person in self.persons:
                if person.label == label:
                    tmp.append(person)
        return tmp

    def create_dir(self, dir_name):
        os.mkdir(os.path.join(self.file_dir, dir_name))

    def create_rct_csv(self, persons=None,
                        cat_roi_prefix='label/catROIs_{}.xml',
                        ct_csv_prefix='roi_ct/{}.csv'):
        if persons is None:
            persons = self.persons
        for person in persons:
            xml_path = os.path.join(self.file_dir,
                                    cat_roi_prefix.format(person.filename))
            if not os.path.exists(xml_path):
                print('No catROIs file:{}:{}'.format(self.name,person.filename))
                continue
            csv_path = os.path.join(self.file_dir,
                                    ct_csv_prefix.format(person.filename))
            report = ET.parse(xml_path)
            root = report.getroot()
            names = root.findall('./aparc_BN_Atlas/names')

            thickness = root.find('./aparc_BN_Atlas/data/thickness')
            thickness = thickness.text.replace(' ', ',')
            thickness = thickness.replace('NaN', '-1')
            
            thickness_list = ast.literal_eval(thickness)

            with open(csv_path, 'w', newline='') as file:
                fieldnames = ['ID', 'CT']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for (item, thickness) in zip(names[0].findall('item'), thickness_list):
                    name = item.text
                    if name[0].lower() == name[-1].lower():
                        if '/' in name:
                            name = name.replace('/', '-')
                        writer.writerow({'ID': name, 'CT': thickness})

    def get_nii_pathes(self, persons=None, nii_prefix='mri/wm{}.nii'):
        pathes = []
        labels = []
        if persons is None:
            persons = self.persons
        for person in persons:
            pathes.append(os.path.join(self.file_dir,
                          nii_prefix.format(person.filename)))
            labels.append(person.label)
        labels = np.asarray(labels)
        return pathes, labels

    def save_nii(self, person, nii, nii_prefix='mri_smoothed_removed/{}.nii'):
        path = os.path.join(self.file_dir,
                            nii_prefix.format(person.filename))
        nifti1.save(nii, path)

    def get_tivs_cgws(self, persons=None, cat_prefix='report/cat_{}.xml'):
        features = []
        labels = []
        if persons is None:
            persons = self.persons
        for person in persons:
            xml_path = os.path.join(self.file_dir,
                                    cat_prefix.format(person.filename))
            report = ET.parse(xml_path)
            root = report.getroot()
            vol_tiv = root.findall('./subjectmeasures/vol_TIV')[0].text
            cgw = root.findall('./subjectmeasures/vol_abs_CGW')
            tmp = [float(i) for i in cgw[0].text.replace('[', '').replace(']', '').split()]
            tmp.insert(0, float(vol_tiv))
            
            features.append(tmp[0:4])
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels

    def get_cortical_thickness(self, persons=None,
                               surf_prefix='surf/s15.mesh.thickness.resampled_32k.{}.gii'):
        features = []
        labels = []
        if persons is None:
            persons = self.persons
        for person in persons:
            ct_path = os.path.join(self.file_dir,
                                   surf_prefix.format(person.filename))
            ct_gii = nib.load(ct_path)
            ct_darray = ct_gii.get_arrays_from_intent(0)[0]
            features.append(ct_darray.data)
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels

    def get_presonal_info_values(self, persons=None,
                                personal_info_prefix='personal_info/{}.csv'):
        # get person's male, female, age, MMSE
        features = []
        labels = []
        if persons is None:
            persons = self.persons
        for person in persons:
            csv_path = os.path.join(self.file_dir,
                                    personal_info_prefix.format(person.filename))
            df = pd.read_csv(csv_path)
            values = df.to_numpy().flatten()
            if len(values) == 3:
                values = np.append(values, np.nan)
            features.append(values)
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels

    def get_csv_values(self, persons=None, prefix='roi_gmv/{}.csv', flatten=False):
        features = []
        labels = []
        if persons is None:
            persons = self.persons
        for person in persons:
            csv_path = os.path.join(self.file_dir,
                                    prefix.format(person.filename))
            df = pd.read_csv(csv_path, index_col=0)
            if flatten:
                features.append(df.to_numpy().flatten())
            else:
                features.append(df.to_numpy())
            labels.append(person.label)
            ids = df.index.tolist()
        features = np.stack(features)
        labels = np.stack(labels)
        return features, labels, ids
    
    def get_csv_df(self, persons=None, prefix='roi_gmv/{}.csv'):
        dfs = []
        labels = []
        if persons is None:
            persons = self.persons
        for person in persons:
            csv_path = os.path.join(self.file_dir,
                                    prefix.format(person.filename))
            df = pd.read_csv(csv_path, index_col=0)
            dfs.append(df)
            labels.append(person.label)
        labels = np.stack(labels)
        return dfs, labels

    def load_label_msn(self, label, _dir='mri_smoothed'):
        path = os.path.join(self.file_dir, _dir, '{}'.format(label))
        mean_path = os.path.join(path, 'mean.nii')
        std_path = os.path.join(path, 'std.nii')
        mean = load_array(mean_path)
        std = load_array(std_path)
        count = len(self.get_by_label(label))
        return mean, std, count