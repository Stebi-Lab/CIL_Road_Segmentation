import os
from typing import Optional, Any

import omegaconf
from omegaconf import OmegaConf

REQUIRED_CONFIG_KEYS = ["name"]
default_config_path = "src/config/default.yaml"
default_config = OmegaConf.load(default_config_path)


def makedirs(path):
    path = path.replace('\\', '/')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def find_file(directory, extension):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)


def verify_config_dict(config) -> None:
    keys = list(config.keys())
    for req in REQUIRED_CONFIG_KEYS:
        if req not in keys:
            raise ValueError("key {} not in config!".format(req))


def merge(to, source, path=[]):
    for key in source.keys():
        if key in to.keys():
            if isinstance(to[key], omegaconf.DictConfig) and isinstance(source[key], omegaconf.DictConfig):
                merge(to[key], source[key], path + [str(key)])
            # elif type(a[key]) is not type(b[key]):
            #     print("ConficlT", key, type(a[key]), type(b[key]))
            #     info = "Conflict at " + (".".join(path + [str(key)]))
            #     raise Exception(info)
        else:
            to[key] = source[key]
    return to


def load_config(config_path: Optional[str] = None) -> Any:
    if config_path is not None:
        config_path = os.path.abspath(config_path)
        # config_dir, config_file = os.path.split(config_path)
        try:
            _config = OmegaConf.load(config_path)
            verify_config_dict(_config.trainer)
            return merge(_config, default_config)
        except ValueError as e:
            raise ValueError(e)
    else:
        _config = default_config
    return _config
