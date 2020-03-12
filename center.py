"""
TODO: add license
"""
import csv
import os
import re

import pandas as pd

import person

class Center(object):
    """docstring for Center"""
    def __init__(self, file_dir, name, filenames,
                 use_nii=False, use_csv=True, 
                 use_personal_info=False, use_xml=True,
                 nii_prefix='mri/{}.nii',
                 csv_prefix='csv/{}.csv',
                 personal_info_prefix='personal_info/{}.csv',
                 xml_prefix='report/cat_{}.xml'):
        super(Center, self).__init__()
        self.file_dir = file_dir
        self.name = name
        self.filenames = filenames
        self.use_nii = use_nii
        self.use_csv = use_csv
        self.use_personal_info = use_personal_info
        self.use_xml = use_xml
        self.nii_prefix = nii_prefix
        self.csv_prefix = csv_prefix
        self.personal_info_prefix = personal_info_prefix
        self.xml_prefix = xml_prefix

class CenterCAT(Center):
    """docstring for Center

    Attributes:
        file_dir: string, center's dir
        filenames: string, filepath to a csv file contains all person's no and their label
        use_nii: bool, whether to load nii
        use_csv: bool, whether to load csv
        use_xml: bool, whether to load xml
        persons: list of Person
    """

    def __init__(self, file_dir, filenames='origin.csv',
                 use_nii=False, use_csv=True, 
                 use_personal_info=False, use_xml=True,
                 nii_prefix='mri/wm{}.nii',
                 csv_prefix='csv/{}.csv',
                 personal_info_prefix='personal_info/{}.csv',
                 xml_prefix='report/cat_{}.xml'):
        name = file_dir[file_dir.rfind('/')+1:]
        super(CenterCAT, self).__init__(file_dir, name, filenames,
                                        use_nii, use_csv, 
                                        use_personal_info, use_xml,
                                        nii_prefix,
                                        csv_prefix, personal_info_prefix, xml_prefix)
        self.use_personal_info = use_personal_info
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
            try:
                filename = index
                label = value['label']
                _person = person.PersonCAT(self.file_dir, filename, label,
                                            use_nii=self.use_nii,
                                            use_csv=self.use_csv,
                                            use_personal_info=self.use_personal_info,
                                            use_xml=self.use_xml,
                                            nii_prefix=self.nii_prefix,
                                            csv_prefix=self.csv_prefix,
                                            personal_info_prefix=self.personal_info_prefix,
                                            xml_prefix=self.xml_prefix)
                persons.append(_person)
            except FileNotFoundError:
                print('File {}/{} Not Found'.format(self.file_dir, filename))
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
            for _person in self.persons:
                if _person.label == label:
                    tmp.append(_person)
        return tmp

    def create_dir(self, dir_name):
        os.mkdir(os.path.join(self.file_dir, dir_name))
