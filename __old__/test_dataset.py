import matplotlib.pyplot as plt

from torch.utils import data
from torchvision import transforms

from dataio.dataset import MICCAIBrats2019Dataset
from dataio.transforms import *


if __name__ == '__main__':

    transform = transforms.Compose([
        RandomIntensityShiftScale(),
        RandomAxisMirrorFlip(),
        RandomCropVolume([128, 128, 128]),
    ])

    dataset = MICCAIBrats2019Dataset(
        root_dir_path='./data/MICCAI_BraTS_2019_Data_Training',
        patient_ids=None,
        transform=transform,
    )

    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        label = sample['label']

        for slice in [20, 60, 120]:

            for k in range(image.shape[0]):
                img = image[k, ..., slice]

                plt.imshow(img, cmap='gray')
                plt.show()
                plt.clf()

            lbl = label[..., slice]

            plt.imshow(lbl, cmap='jet')
            plt.show()
            plt.clf()
