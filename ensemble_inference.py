import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import nibabel as nib
import itertools
from functools import partial
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataio import get_data_loader
from networks import ResUNet
from networks import init_weights
from functions import OneHotEncoder
from functions import SoftDiceLoss
from functions import FocalLoss
from functions import ActiveContourLoss
from functions import DiceCoefficient
from utils import load_json
from utils import check_manual_seed
from utils import get_output_dir_path
from utils import save_config
from utils import save_logs
from utils import print_times
from utils import save_images_via_plt


def get_trained_model(model_config):
    model = ResUNet(
        input_dim=model_config.input_dim,
        output_dim=model_config.output_dim,
        filters=model_config.filters,
    )

    print(model)

    print('Loading saved model: {}'.format(model_config.saved_model))
    model.load_state_dict(torch.load(model_config.saved_model)['model'])

    if config.run.use_cuda:
        model.cuda()
        model = nn.DataParallel(model)

    return model


def inference(config):

    if config.run.visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices

    test_patient_ids = os.listdir(config.test_dataset.root_dir_path)

    seed = check_manual_seed()
    print('Using seed: {}'.format(seed))

    class_name_to_index = config.label_to_id._asdict()
    index_to_class_name = {v: k for k, v in class_name_to_index.items()}

    test_data_loader = get_data_loader(
        mode='test',
        dataset_name=config.test_dataset.dataset_name,
        root_dir_path=config.test_dataset.root_dir_path,
        patient_ids=test_patient_ids,
        batch_size=config.test_dataset.batch_size,
        num_workers=config.test_dataset.num_workers,
        volume_size=config.test_dataset.volume_size,
    )

    model_1 = get_trained_model(config.model_1)
    model_2 = get_trained_model(config.model_2)
    model_3 = get_trained_model(config.model_3)
    model_4 = get_trained_model(config.model_4)
    model_5 = get_trained_model(config.model_5)

    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()

    for batch in tqdm(test_data_loader):
        image = batch['image'].cuda().float()
        assert image.size(0) == 1

        patient_id = batch['patient_id'][0]
        nii_path = batch['nii_path'][0]

        image = F.pad(image, (2, 3, 0, 0, 0, 0, 0, 0, 0, 0), 'constant', 0)
        output = torch.ones((1, image.shape[2], image.shape[3], image.shape[4]))

        with torch.no_grad():
            pred_1 = model_1(image)
            pred_2 = model_2(image)
            pred_3 = model_3(image)
            pred_4 = model_4(image)
            pred_5 = model_5(image)

            pred = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5) / 5.0

        mask_0 = pred[:, 0, ...] < 0.5
        mask_1 = pred[:, 1, ...] < 0.5
        mask_2 = pred[:, 2, ...] < 0.5
        mask = mask_0 * mask_1 * mask_2

        pred = pred.argmax(1).cpu()
        output += pred

        output[mask] = 0

        image = image[..., 2:-3]
        output = output[..., 2:-3]

        save_dir_path = os.path.join(config.save.save_root_dir, patient_id)
        os.makedirs(save_dir_path, exist_ok=True)

        image = image.cpu().numpy()[0, 1, ...]
        output = output.cpu().numpy()[0, ...].astype(np.int16)

        nii_image = nib.load(nii_path)

        nii_output = nib.Nifti1Image(output, affine=nii_image.affine)

        nib.save(nii_output, os.path.join(os.path.join(
            save_dir_path, patient_id + '_output.nii.gz'))
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmentation boilerplate')
    parser.add_argument('-c', '--config', help='config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)

    inference(config)
