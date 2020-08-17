from torch.utils import data
from torchvision import transforms

from dataio.dataset import MICCAIBrats2019Dataset
from dataio.transforms import ToTensor
from dataio.transforms import RandomIntensityShiftScale
from dataio.transforms import RandomAxisMirrorFlip
from dataio.transforms import RandomCropVolume
from dataio.transforms import StaticCropVolume


def get_data_loader(mode, dataset_name, root_dir_path, patient_ids,
                    batch_size, num_workers, volume_size):

    assert mode in ['train', 'val', 'test']
    assert dataset_name == 'MICCAIBrats2019Dataset'

    if mode == 'train':
        TRANSFORM = [
            RandomIntensityShiftScale(),
            RandomAxisMirrorFlip(),
            RandomCropVolume(volume_size),
            ToTensor(),
        ]
        shuffle = True

    elif mode == 'val':
        TRANSFORM = [
            StaticCropVolume(volume_size),
            ToTensor(),
        ]
        shuffle = False

    elif mode == 'test':
        TRANSFORM = [
            ToTensor(),
        ]
        shuffle = False

    dataset = MICCAIBrats2019Dataset(
        root_dir_path=root_dir_path,
        patient_ids=patient_ids,
        transform=transforms.Compose(TRANSFORM),
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
