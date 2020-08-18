import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt


config = {
    'root_dir_path': './data/MICCAI_BraTS_2019_Data_Training',
    'modalities': [
        {'name': 'T1', 'pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg'},
    ],
}


def z_score_normalize(array):
    array = array.astype(np.float32)
    mask = array > 0

    mean = np.mean(array[mask])
    std = np.std(array[mask])

    array -= mean
    array /= std

    return array


if __name__ == '__main__':

    for patient_id in tqdm(sorted(os.listdir(config['root_dir_path']))):
        print(patient_id)
        patient_dir_path = os.path.join(
            config['root_dir_path'], patient_id,
        )

        for modality in config['modalities']:
            file_path = os.path.join(
                patient_dir_path,
                patient_id + '_' + modality['pattern'] + '.nii.gz'
            )
            nii_file = nib.load(file_path)
            series = nii_file.get_data()

            if modality['name'] == 'SEG':
                series = series.astype(np.int32)
                bincount = np.bincount(series.ravel())
                if len(bincount) > 3:
                    assert bincount[3] == 0

                series[series == 4] = 3  # 3: ET (GD-enhancing tumor)
                series[series == 2] = 2  # 2: ED (peritumoral edema)
                series[series == 1] = 1  # 1: NCR/NET (non-enhancing tumor core)
                series[series == 0] = 0  # 0: Background

            else:
                series = z_score_normalize(series)

            series = nib.Nifti1Image(series, nii_file.affine)

            save_path = os.path.join(
                patient_dir_path,
                patient_id + '_' + modality['pattern'] + '_norm.nii.gz'
            )

            nib.save(series, save_path)
