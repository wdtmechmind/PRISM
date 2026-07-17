import os

import yaml


def load_yaml_config(path):
    if not path:
        return {}

    config_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError('config must be a YAML mapping: %s' % config_path)
    return data


def merge_defaults(base_defaults, config_values):
    merged = dict(base_defaults)
    for key, value in config_values.items():
        if key not in merged:
            raise ValueError('unknown config key: %s' % key)
        merged[key] = value
    return merged