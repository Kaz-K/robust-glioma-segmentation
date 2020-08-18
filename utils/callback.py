import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def save_config(engine, config, seed=None, output_dir_path=None):
    config_to_save = defaultdict(dict)
    for key, child in config._asdict().items():
        for k, v in child._asdict().items():
            config_to_save[key][k] = v

    if seed:
        config_to_save['seed'] = seed
    if output_dir_path:
        config_to_save['output_dir_path'] = output_dir_path

    save_path = os.path.join(output_dir_path, 'config.json')
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f)


def save_logs(mode, engine, config, output_dir_path):
    if mode == 'train':
        fname = os.path.join(output_dir_path, 'train_logs.csv')
    elif mode == 'val':
        fname = os.path.join(output_dir_path, 'val_logs.csv')
    elif mode == 'precision_val':
        fname = os.path.join(output_dir_path, 'precision_val_logs.csv')

    columns = ['epoch', 'iteration'] + list(engine.state.metrics.keys())
    values = [str(engine.state.epoch), str(engine.state.iteration)] \
           + [str(value) for value in engine.state.metrics.values()]

    with open(fname, 'a') as f:
        if f.tell() == 0:
            print(','.join(columns), file=f)
        print(','.join(values), file=f)


def print_times(engine, config, pbar, timer):
    pbar.log_message(
        'Epoch {} done. Time per batch: {:.3f}[s]'.format(
            engine.state.epoch, timer.value()
        )
    )
    timer.reset()


def save_images_via_plt(image, label, output, n_save_images, config, save_path):
    n_columns = min(image.shape[0], n_save_images)
    for i in range(1, n_columns + 1):
        plt.subplot(n_columns, 3, 1 + 3 * (i - 1))
        plt.imshow(image[i - 1, ...], vmin=0, vmax=255, cmap='gray')
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.subplot(n_columns, 3, 2 + 3 * (i - 1))
        plt.imshow(label[i - 1, ...], vmin=config.save.label_vmin, vmax=config.save.label_vmax, cmap='jet')
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.subplot(n_columns, 3, 3 + 3 * (i - 1))
        plt.imshow(output[i - 1, ...], vmin=config.save.label_vmin, vmax=config.save.label_vmax, cmap='jet')
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
