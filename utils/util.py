import os
import json
import random
import collections
import torch
import datetime


def load_json(path):
    def _json_object_hook(d):
        for k, v in d.items():
            d[k] = None if v is False else v
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())


def check_manual_seed(seed=None):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_output_dir_path(config, i=None):
    study_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

    if i is None:
        dir_name = config.save.study_name + '_' + study_time
    else:
        dir_name = config.save.study_name + '_' + str(i) + '_' + study_time

    output_dir_path = os.path.join(
        config.save.output_root_dir, dir_name
    )
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path
