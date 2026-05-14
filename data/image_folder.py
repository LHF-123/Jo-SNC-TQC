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
        self.tqc_group_summary = None
        self.tqc_soft_labels = None
        if tqc_group_path is not None:
            self.tqc_groups, self.tqc_group_names, self.tqc_group_summary = self._load_tqc_groups(tqc_group_path)
        if tqc_soft_label_path is not None:
            self.tqc_soft_labels = np.load(tqc_soft_label_path).astype(np.float32)
            assert len(self.tqc_soft_labels) == len(self.samples), \
                f'TQC soft labels length {len(self.tqc_soft_labels)} does not match dataset length {len(self.samples)}'

    @staticmethod
    def _tqc_image_key(path):
        parts = os.path.normpath(path).replace('\\', '/').split('/')
        if 'train' in parts:
            train_pos = len(parts) - 1 - parts[::-1].index('train')
            return '/'.join(parts[train_pos + 1:])
        return '/'.join(parts[-2:])

    @staticmethod
    def _tqc_group_id(record):
        return int(record.get('train_group_id', record.get('group_id', 0)))

    @staticmethod
    def _summarize_tqc_groups(groups, num_dataset_samples):
        anchor_count = int((groups == 1).sum())
        sibling_count = int((groups == 2).sum())
        ignored_count = int((groups == 0).sum())
        return {
            'num_groups': int(len(groups)),
            'num_dataset_samples': int(num_dataset_samples),
            'anchor_count': anchor_count,
            'sibling_count': sibling_count,
            'ignored_count': ignored_count,
        }

    def _load_tqc_groups(self, tqc_group_path):
        assert os.path.isfile(tqc_group_path), f'TQC group file does not exist: {tqc_group_path}'
        with open(tqc_group_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and 'samples' in payload:
            sample_records = payload['samples']
        elif isinstance(payload, list):
            sample_records = payload
        else:
            raise AssertionError(f'Unsupported TQC group file format: {tqc_group_path}')

        if len(sample_records) > 0 and all(isinstance(record, dict) and 'image_path' in record for record in sample_records):
            groups = np.zeros(len(self.samples), dtype=np.int64)
            group_names = ['ignore'] * len(self.samples)
            records_by_key = {}
            duplicate_keys = []
            for record in sample_records:
                key = self._tqc_image_key(record['image_path'])
                if key in records_by_key:
                    duplicate_keys.append(key)
                records_by_key[key] = record
            if duplicate_keys:
                raise AssertionError(f'Duplicate image_path key(s) in TQC group file: {duplicate_keys[:5]}')

            missing_keys = []
            label_mismatches = []
            used_keys = set()
            for idx, sample in enumerate(self.samples):
                path, target = sample
                key = self._tqc_image_key(path)
                record = records_by_key.get(key)
                if record is None:
                    missing_keys.append(key)
                    continue
                used_keys.add(key)
                if 'label' in record and int(record['label']) != int(target):
                    label_mismatches.append((key, int(record['label']), int(target)))
                groups[idx] = self._tqc_group_id(record)
                group_names[idx] = record.get('group', 'ignore')

            extra_keys = sorted(set(records_by_key.keys()) - used_keys)
            if missing_keys or extra_keys or label_mismatches:
                raise AssertionError(
                    'TQC group file does not align with dataset samples. '
                    f'missing={missing_keys[:5]}, extra={extra_keys[:5]}, label_mismatches={label_mismatches[:5]}'
                )
        else:
            groups = np.zeros(len(sample_records), dtype=np.int64)
            group_names = ['ignore'] * len(sample_records)
            seen_indices = set()
            for fallback_idx, record in enumerate(sample_records):
                idx = int(record.get('index', fallback_idx))
                if idx < 0 or idx >= len(groups):
                    raise AssertionError(f'TQC group index out of range: index={idx}, length={len(groups)}')
                if idx in seen_indices:
                    raise AssertionError(f'Duplicate TQC group index in fallback format: {idx}')
                seen_indices.add(idx)
                groups[idx] = self._tqc_group_id(record)
                group_names[idx] = record.get('group', 'ignore')

        assert len(groups) == len(self.samples), \
            f'TQC group length {len(groups)} does not match dataset length {len(self.samples)}'
        return groups, group_names, self._summarize_tqc_groups(groups, len(self.samples))

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

