import os
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data
import nibabel as nib


class MICCAIBrats2019Dataset(data.Dataset):
    modalities = [
        {'name': 'T1', 'pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg'},
    ]

    def __init__(self, root_dir_path, patient_ids, transform):
        self.root_dir_path = root_dir_path
        self.patient_ids = patient_ids if patient_ids is not None else os.listdir(root_dir_path)
        self.transform = transform

        self.build_file_paths()

    def build_file_paths(self):
        self.files = []
        for patient_id in self.patient_ids:
            patient_dir_path = os.path.join(self.root_dir_path, patient_id)

            file_paths = {}
            for modality in self.modalities:
                file_path = os.path.join(
                    patient_dir_path,
                    patient_id + '_' + modality['pattern'] + '_norm.nii.gz'
                )
                assert os.path.exists(file_path)

                file_paths.update({
                    modality['name']: file_path
                })

            self.files.append(file_paths)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]

        image = None
        label = None

        for modality in item.keys():
            if modality != 'SEG':
                series = nib.load(item[modality]).get_data()
                if image is None:
                    image = series[np.newaxis, ...]
                else:
                    image = np.concatenate((image, series[np.newaxis, ...]), axis=0)

            else:
                label = nib.load(item[modality]).get_data()

        sample = {
            'image': image.astype(np.float32),
            'label': label.astype(np.int32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
