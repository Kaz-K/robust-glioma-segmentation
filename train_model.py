import os
import random
import argparse
import numpy as np
from functools import partial
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

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


def divide_patients(root_dir_path, ratio=0.8):
    random.seed(0)
    all_patients = os.listdir(root_dir_path)
    random.shuffle(all_patients)
    n_train = int(len(all_patients) * ratio)
    train_patient_ids = all_patients[:n_train]
    val_patient_ids = all_patients[n_train:]
    return train_patient_ids, val_patient_ids


def get_cv_splits(root_dir_path, i):
    all_patients = os.listdir(root_dir_path)
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    train_index, val_index = list(kf.split(all_patients))[i]
    train_patient_ids = [all_patients[i] for i in list(train_index)]
    val_patient_ids = [all_patients[i] for i in list(val_index)]
    return train_patient_ids, val_patient_ids


def adjust_learning_rate(optimizer, epoch, initial_lr, n_epochs, gamma=0.9):
    lr = initial_lr * (1 - (epoch / n_epochs)) ** gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(config, needs_save, i):

    if config.run.visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices

    assert config.train_dataset.root_dir_path == config.val_dataset.root_dir_path
    # train_patient_ids, val_patient_ids = divide_patients(config.train_dataset.root_dir_path)
    train_patient_ids, val_patient_ids = get_cv_splits(config.train_dataset.root_dir_path, i)

    seed = check_manual_seed()
    print('Using seed: {}'.format(seed))

    class_name_to_index = config.label_to_id._asdict()
    index_to_class_name = {v: k for k, v in class_name_to_index.items()}

    train_data_loader = get_data_loader(
        mode='train',
        dataset_name=config.train_dataset.dataset_name,
        root_dir_path=config.train_dataset.root_dir_path,
        patient_ids=train_patient_ids,
        batch_size=config.train_dataset.batch_size,
        num_workers=config.train_dataset.num_workers,
        volume_size=config.train_dataset.volume_size,
    )

    val_data_loader = get_data_loader(
        mode='val',
        dataset_name=config.val_dataset.dataset_name,
        root_dir_path=config.val_dataset.root_dir_path,
        patient_ids=val_patient_ids,
        batch_size=config.val_dataset.batch_size,
        num_workers=config.val_dataset.num_workers,
        volume_size=config.val_dataset.volume_size,
    )

    model = ResUNet(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        filters=config.model.filters,
    )

    print(model)

    if config.run.use_cuda:
        model.cuda()
        model = nn.DataParallel(model)

    if config.model.saved_model:
        print('Loading saved model: {}'.format(config.model.saved_model))
        model.load_state_dict(torch.load(config.model.saved_model))
    else:
        print('Initializing weights.')
        init_weights(model, init_type=config.model.init_type)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.optimizer.lr,
                           betas=config.optimizer.betas,
                           weight_decay=config.optimizer.weight_decay)

    dice_loss = SoftDiceLoss()

    focal_loss = FocalLoss(
        gamma=config.focal_loss.gamma,
        alpha=config.focal_loss.alpha,
    )

    active_contour_loss = ActiveContourLoss(
        weight=config.active_contour_loss.weight,
    )

    dice_coeff = DiceCoefficient(
        n_classes=config.metric.n_classes,
        index_to_class_name=index_to_class_name,
    )

    one_hot_encoder = OneHotEncoder(
        n_classes=config.metric.n_classes,
    ).forward

    def train(engine, batch):
        adjust_learning_rate(optimizer,
                             engine.state.epoch,
                             initial_lr=config.optimizer.lr,
                             n_epochs=config.run.n_epochs,
                             gamma=config.optimizer.gamma)

        model.train()

        image = batch['image']
        label = batch['label']

        if config.run.use_cuda:
            image = image.cuda(non_blocking=True).float()
            label = label.cuda(non_blocking=True).long()

        else:
            image = image.float()
            label = label.long()

        optimizer.zero_grad()

        output = model(image)
        target = one_hot_encoder(label)[:, 1:, ...]

        l_dice = dice_loss(output, target)
        l_focal = focal_loss(output, target)
        l_active_contour = active_contour_loss(output, target)

        l_total = l_dice + l_focal + l_active_contour
        l_total.backward()

        optimizer.step()

        m_dice = dice_coeff.update(output.detach(), label)

        measures = {
            'SoftDiceLoss': l_dice.item(),
            'FocalLoss': l_focal.item(),
            'ActiveContourLoss': l_active_contour.item(),
        }

        measures.update(m_dice)

        if config.run.use_cuda:
            torch.cuda.synchronize()

        return measures

    def evaluate(engine, batch):
        model.eval()

        image = batch['image']
        label = batch['label']

        if config.run.use_cuda:
            image = image.cuda(non_blocking=True).float()
            label = label.cuda(non_blocking=True).long()

        else:
            image = image.float()
            label = label.long()

        with torch.no_grad():
            output = model(image)
            target = one_hot_encoder(label)[:, 1:, ...]

            l_dice = dice_loss(output, target)
            l_focal = focal_loss(output, target)
            l_active_contour = active_contour_loss(output, target)

            m_dice = dice_coeff.update(output.detach(), label)

        measures = {
            'SoftDiceLoss': l_dice.item(),
            'FocalLoss': l_focal.item(),
            'ActiveContourLoss': l_active_contour.item(),
        }

        measures.update(m_dice)

        if config.run.use_cuda:
            torch.cuda.synchronize()

        return measures

    output_dir_path = get_output_dir_path(config, i)
    trainer = Engine(train)
    evaluator = Engine(evaluate)
    timer = Timer(average=True)

    if needs_save:
        checkpoint_handler = ModelCheckpoint(
            output_dir_path,
            config.save.study_name,
            save_interval=config.save.save_epoch_interval,
            n_saved=config.run.n_epochs + 1,
            create_dir=True,
        )

    monitoring_metrics = ['SoftDiceLoss', 'FocalLoss', 'ActiveContourLoss']
    monitoring_metrics += class_name_to_index.keys()

    for metric in monitoring_metrics:
        RunningAverage(
            alpha=0.98,
            output_transform=partial(lambda x, metric: x[metric], metric=metric)
        ).attach(trainer, metric)

    for metric in monitoring_metrics:
        RunningAverage(
            alpha=0.98,
            output_transform=partial(lambda x, metric: x[metric], metric=metric)
        ).attach(evaluator, metric)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=monitoring_metrics)
    pbar.attach(evaluator, metric_names=monitoring_metrics)

    @trainer.on(Events.STARTED)
    def call_save_config(engine):
        if needs_save:
            return save_config(engine, config, seed, output_dir_path)

    @trainer.on(Events.EPOCH_COMPLETED)
    def call_save_logs(engine):
        if needs_save:
            return save_logs('train', engine, config, output_dir_path)

    @trainer.on(Events.EPOCH_COMPLETED)
    def call_print_times(engine):
        return print_times(engine, config, pbar, timer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        evaluator.run(val_data_loader, 1)

        if needs_save:
            save_logs('val', evaluator, config, output_dir_path)
            save_images(evaluator, trainer.state.epoch)

    def save_images(evaluator, epoch):
        batch = evaluator.state.batch
        image = batch['image']
        label = batch['label']

        if config.run.use_cuda:
            image = image.cuda(non_blocking=True).float()
            label = label.cuda(non_blocking=True).long()
        else:
            image = image.float()
            label = label.long()

        with torch.no_grad():
            pred = model(image)

        output = torch.ones_like(label)

        mask_0 = pred[:, 0, ...] < 0.5
        mask_1 = pred[:, 1, ...] < 0.5
        mask_2 = pred[:, 2, ...] < 0.5
        mask = mask_0 * mask_1 * mask_2

        pred = pred.argmax(1)
        output += pred

        output[mask] = 0

        image = image.detach().cpu().float()
        label = label.detach().cpu().unsqueeze(1).float()
        output = output.detach().cpu().unsqueeze(1).float()

        z_middle = image.shape[-1] // 2
        image = image[:, 0, ..., z_middle]
        label = label[:, 0, ..., z_middle]
        output = output[:, 0, ..., z_middle]

        if config.save.image_vmax is not None:
            vmax = config.save.image_vmax
        else:
            vmax = image.max()

        if config.save.image_vmin is not None:
            vmin = config.save.image_vmin
        else:
            vmin = image.min()

        image = np.clip(image, vmin, vmax)
        image -= vmin
        image /= (vmax - vmin)
        image *= 255.0

        save_path = os.path.join(
            output_dir_path, 'result_{}.png'.format(epoch)
        )
        save_images_via_plt(image, label, output, config.save.n_save_images, config, save_path)

    if needs_save:
        trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                                  handler=checkpoint_handler,
                                  to_save={'model': model, 'optim': optimizer})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.run.n_epochs, config.run.n_epochs * len(train_data_loader))
    )

    trainer.run(train_data_loader, config.run.n_epochs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmentation boilerplate')
    parser.add_argument('-c', '--config', help='config file', required=True)
    parser.add_argument('-s', '--save', help='save logs', action='store_true')
    parser.add_argument('-i', '--i', help='i-th hold for 5-cv', default=0)
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save, int(args.i))
