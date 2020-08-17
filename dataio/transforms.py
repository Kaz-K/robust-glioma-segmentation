import torch
import random
import numpy as np


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        if image.ndim == 3:
            image = image[np.newaxis, ...]

        sample.update({
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).int(),
        })

        return sample


class RandomIntensityShiftScale(object):
    def __call__(self, sample):
        image = sample['image']

        for i in range(image.shape[0]):
            shift = random.uniform(-0.1, 0.1)
            scale = random.uniform(0.9, 1.1)
            img = image[i, ...]
            image[i, ...] = scale * (img + shift)

        sample.update({
            'image': image,
        })

        return sample


class RandomAxisMirrorFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        for i in range(1, image.ndim):
            if random.uniform(0, 1) < 0.5:
                image = np.flip(image, axis=i)
                label = np.flip(label, axis=i-1)

        sample.update({
            'image': image,
            'label': label,
        })

        return sample


class RandomCropVolume(object):
    def __init__(self, volume_size):
        self.volume_size = volume_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        assert image.shape[1:] == label.shape

        x_src, y_src, z_src = image.shape[1:]
        x_dst, y_dst, z_dst = self.volume_size

        assert x_src >= x_dst
        assert y_src >= y_dst
        assert z_src >= z_dst

        xs = random.randint(0, x_src - x_dst)
        ys = random.randint(0, y_src - y_dst)
        zs = random.randint(0, z_src - z_dst)

        image = image[:,
                      xs: xs + x_dst,
                      ys: ys + y_dst,
                      zs: zs + z_dst]

        label = label[xs: xs + x_dst,
                      ys: ys + y_dst,
                      zs: zs + z_dst]

        sample.update({
            'image': image,
            'label': label,
        })

        return sample
