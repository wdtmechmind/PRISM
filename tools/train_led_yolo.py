#!/usr/bin/env python3
"""Prepare and train a YOLO detector for PRISM LED annotation data.

The script expects the dataset layout produced by tools/annotate_led_dataset.py:

    <dataset-root>/images/cam0/set_000001.png
    <dataset-root>/images/cam1/set_000001.png
    ...
    <dataset-root>/labels/cam0/set_000001.txt
    <dataset-root>/labels/cam1/set_000001.txt
    ...

It converts that layout into a standard Ultralytics dataset structure with a
train/val split, then launches training.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

CLASS_NAMES = ['red', 'yellow', 'blue', 'green']


def _safe_symlink_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.abspath(src), dst)
    except Exception:
        shutil.copy2(src, dst)


def _list_set_ids(dataset_root):
    images_root = dataset_root / 'images'
    labels_root = dataset_root / 'labels'
    if not images_root.is_dir():
        raise RuntimeError('missing images directory: %s' % images_root)
    if not labels_root.is_dir():
        raise RuntimeError('missing labels directory: %s' % labels_root)

    cam_dirs = sorted(path for path in images_root.iterdir() if path.is_dir())
    if len(cam_dirs) != 4:
        raise RuntimeError('expected 4 camera image directories under %s, found %d' % (images_root, len(cam_dirs)))

    set_ids = None
    samples = []
    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        label_dir = labels_root / cam_name
        if not label_dir.is_dir():
            raise RuntimeError('missing label directory: %s' % label_dir)

        cam_sets = set()
        for image_path in sorted(cam_dir.glob('*.png')):
            stem = image_path.stem
            label_path = label_dir / (stem + '.txt')
            if not label_path.is_file():
                raise RuntimeError('missing label file for %s: %s' % (image_path, label_path))
            cam_sets.add(stem)
        if set_ids is None:
            set_ids = cam_sets
        else:
            set_ids &= cam_sets

    if not set_ids:
        raise RuntimeError('no complete synchronized frame sets found under %s' % dataset_root)

    for stem in sorted(set_ids):
        sample = {'set_id': stem, 'cams': []}
        missing = False
        for cam_dir in cam_dirs:
            cam_name = cam_dir.name
            image_path = images_root / cam_name / (stem + '.png')
            label_path = labels_root / cam_name / (stem + '.txt')
            if not image_path.is_file() or not label_path.is_file():
                missing = True
                break
            sample['cams'].append({'cam': cam_name, 'image': image_path, 'label': label_path})
        if not missing:
            samples.append(sample)

    if not samples:
        raise RuntimeError('no usable synchronized samples found under %s' % dataset_root)
    return samples


def _split_samples(samples, val_fraction, seed):
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(val_fraction)))) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    if not train_samples:
        train_samples, val_samples = shuffled[:-1], shuffled[-1:]
    return train_samples, val_samples


def _materialize_split(samples, split_name, prepared_root):
    image_dir = prepared_root / 'images' / split_name
    label_dir = prepared_root / 'labels' / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for sample in samples:
        set_id = sample['set_id']
        for cam_entry in sample['cams']:
            cam_name = cam_entry['cam']
            stem = '%s__%s' % (set_id, cam_name)
            _safe_symlink_or_copy(cam_entry['image'], image_dir / (stem + '.png'))
            _safe_symlink_or_copy(cam_entry['label'], label_dir / (stem + '.txt'))
            total += 1
    return total


def _write_dataset_yaml(prepared_root, yaml_path):
    payload = {
        'path': str(prepared_root),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return payload


def _print_summary(samples, train_samples, val_samples, prepared_root, yaml_path):
    summary = {
        'source_sets': len(samples),
        'train_sets': len(train_samples),
        'val_sets': len(val_samples),
        'prepared_root': str(prepared_root),
        'dataset_yaml': str(yaml_path),
        'classes': CLASS_NAMES,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(description='Prepare and train YOLO on PRISM LED annotations')
    p.add_argument('--dataset-root', type=str, default='data/datasets/led_yolo',
                   help='root produced by tools/annotate_led_dataset.py')
    p.add_argument('--prepared-root', type=str, default='',
                   help='output directory for the YOLO-formatted split dataset')
    p.add_argument('--yaml-path', type=str, default='',
                   help='output dataset YAML path; defaults to <prepared-root>/dataset.yaml')
    p.add_argument('--val-fraction', type=float, default=0.2, help='fraction of synchronized sets used for validation')
    p.add_argument('--seed', type=int, default=42, help='random seed for the train/val split')
    p.add_argument('--weights', type=str, default='yolov8n.pt', help='initial YOLO weights or model name')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--name', type=str, default='prism_led_yolo')
    p.add_argument('--project', type=str, default='runs/detect')
    p.add_argument('--exist-ok', action='store_true', default=True)
    p.add_argument('--no-train', action='store_true', help='only prepare the dataset split, do not train')
    p.add_argument('--cache', action='store_true', help='enable Ultralytics cache')
    p.add_argument('--pretrained', action='store_true', default=True, help='use pretrained initialization if available')
    return p.parse_args()


def main(argv=None):
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise RuntimeError('dataset root not found: %s' % dataset_root)

    prepared_root = Path(args.prepared_root).expanduser().resolve() if args.prepared_root else (dataset_root / '_prepared_yolo')
    yaml_path = Path(args.yaml_path).expanduser().resolve() if args.yaml_path else (prepared_root / 'dataset.yaml')

    samples = _list_set_ids(dataset_root)
    train_samples, val_samples = _split_samples(samples, args.val_fraction, args.seed)

    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    train_count = _materialize_split(train_samples, 'train', prepared_root)
    val_count = _materialize_split(val_samples, 'val', prepared_root)
    _write_dataset_yaml(prepared_root, yaml_path)
    _print_summary(samples, train_samples, val_samples, prepared_root, yaml_path)
    print('prepared images: train=%d val=%d' % (train_count, val_count))

    if args.no_train:
        return

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            'ultralytics is required for training. Install it first, for example:\n'
            '  pip install ultralytics'
        ) from exc

    model = YOLO(args.weights)
    train_kwargs = {
        'data': str(yaml_path),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'cache': args.cache,
        'pretrained': args.pretrained,
    }
    if args.device:
        train_kwargs['device'] = args.device

    print('starting training with %s' % json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    model.train(**train_kwargs)


if __name__ == '__main__':
    main()