# -*- coding: utf-8 -*-

from torchvision.datasets import DatasetFolder
from PIL import Image
from tqdm import tqdm
import json
import os
import numpy as np


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class IndexedImageFolder(DatasetFolder):
    def __init__(self, root, use_cache=False, transform=None, target_transform=None,
                 loader=pil_loader, is_valid_file=None, tqc_group_path=None, tqc_soft_label_path=None):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        self.imgs = self.samples  # list, element is (path, label)

        self.use_cache = use_cache
        if self.use_cache:
            self.loaded_samples = self._cache_dataset()  # list, element is (PIL image, label)
        else:
            self.loaded_samples = None

        self.tqc_groups = None
        self.tqc_group_names = None
        self.tqc_soft_labels = None
        if tqc_group_path is not None:
            self.tqc_groups, self.tqc_group_names = self._load_tqc_groups(tqc_group_path)
        if tqc_soft_label_path is not None:
            self.tqc_soft_labels = np.load(tqc_soft_label_path).astype(np.float32)
            assert len(self.tqc_soft_labels) == len(self.samples), \
                f'TQC soft labels length {len(self.tqc_soft_labels)} does not match dataset length {len(self.samples)}'

    def _load_tqc_groups(self, tqc_group_path):
        assert os.path.isfile(tqc_group_path), f'TQC group file does not exist: {tqc_group_path}'
        with open(tqc_group_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and 'samples' in payload:
            sample_records = payload['samples']
            groups = np.zeros(len(sample_records), dtype=np.int64)
            group_names = ['ignore'] * len(sample_records)
            for record in sample_records:
                idx = int(record['index'])
                groups[idx] = int(record.get('train_group_id', record.get('group_id', 0)))
                group_names[idx] = record.get('group', 'ignore')
        elif isinstance(payload, list):
            groups = np.zeros(len(payload), dtype=np.int64)
            group_names = ['ignore'] * len(payload)
            for idx, record in enumerate(payload):
                groups[idx] = int(record.get('train_group_id', record.get('group_id', 0)))
                group_names[idx] = record.get('group', 'ignore')
        else:
            raise AssertionError(f'Unsupported TQC group file format: {tqc_group_path}')
        assert len(groups) == len(self.samples), \
            f'TQC group length {len(groups)} does not match dataset length {len(self.samples)}'
        return groups, group_names

    def _cache_dataset(self):
        cached_dataset = []
        n_samples = len(self.samples)
        print('caching samples ... ')
        for idx, sample in enumerate(tqdm(self.samples, ncols=100, ascii=' >')):
            path, target = sample
            image = self.loader(path)
            item = (image, target)
            cached_dataset.append(item)
        assert len(cached_dataset) == n_samples
        return cached_dataset

    def __getitem__(self, index):
        if self.use_cache:
            assert len(self.loaded_samples) == len(self.samples)
            sample, target = self.loaded_samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        item = {'index': index, 'data': sample, 'label': target}
        if self.tqc_groups is not None:
            item['group'] = int(self.tqc_groups[index])
            item['group_name'] = self.tqc_group_names[index]
        if self.tqc_soft_labels is not None:
            item['soft_label'] = self.tqc_soft_labels[index]
        return item

