# -*- coding: utf-8 -*-
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_CLASS_PROMPTS = [
    'a photo of a {class_name}',
    'a close-up photo of a {class_name}',
    'a fine-grained photo of a {class_name}',
]

DEFAULT_OUT_DOMAIN_PROMPTS = [
    'a photo of a plant',
    'a photo of a person',
    'a photo of a landscape',
    'a photo of text',
    'a photo of a logo',
    'a photo of an indoor object',
    'a photo of a building',
]

DOMAIN_PROMPTS = {
    'web-bird': 'a photo of a bird',
    'web-car': 'a photo of a car',
    'web-aircraft': 'a photo of an aircraft',
}


class ImagePathDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        bad_image = False
        try:
            with Image.open(path) as image_file:
                image = image_file.convert('RGB')
        except Exception as exc:
            bad_image = True
            image = Image.new('RGB', (224, 224), 'black')
            print(f'[Warning] failed to read image, using black placeholder: index={index}, path={path}, error={exc}', flush=True)
        return {
            'index': index,
            'image': self.preprocess(image),
            'label': label,
            'path': path,
            'bad_image': bad_image,
        }


def load_clip_backend(model_name, device):
    try:
        import clip
        model, preprocess = clip.load(model_name, device=device)

        def tokenize(texts):
            return clip.tokenize(texts).to(device)

        return model.eval(), preprocess, tokenize
    except ImportError:
        pass

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            'Install either OpenAI CLIP (`pip install git+https://github.com/openai/CLIP.git`) '
            'or open_clip_torch before running this script.'
        ) from exc

    open_clip_name = model_name.replace('/', '-')
    model, _, preprocess = open_clip.create_model_and_transforms(open_clip_name, pretrained='openai', device=device)
    tokenizer = open_clip.get_tokenizer(open_clip_name)

    def tokenize(texts):
        return tokenizer(texts).to(device)

    return model.eval(), preprocess, tokenize


def encode_texts(model, tokenize, texts, device, batch_size):
    feats = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenize(batch)
            text_features = model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=1)
            feats.append(text_features.cpu())
    return torch.cat(feats, dim=0).numpy()


def encode_images(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = []
    bad_records = []
    with torch.no_grad():
        for sample in tqdm(loader, ncols=100, ascii=' >', desc='CLIP image features'):
            images = sample['image'].to(device)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features.float(), dim=1)
            feats.append(image_features.cpu())
            bad_flags = sample.get('bad_image', None)
            if bad_flags is not None:
                bad_flags = bad_flags.detach().cpu().numpy().astype(bool)
                indices = sample['index'].detach().cpu().numpy().tolist()
                labels = sample['label'].detach().cpu().numpy().tolist()
                paths = sample['path']
                for flag, index, label, path in zip(bad_flags, indices, labels, paths):
                    if flag:
                        bad_records.append((index, path, label))
    return torch.cat(feats, dim=0).numpy(), bad_records


def write_samples_csv(path, samples, classes):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'image_path', 'label', 'class_name'])
        for idx, (image_path, label) in enumerate(samples):
            writer.writerow([idx, image_path, label, classes[label]])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='Dataset root, e.g. ../datasets/web-bird')
    parser.add_argument('--dataset', required=True, choices=['web-bird', 'web-car', 'web-aircraft'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--clip-model', default='ViT-B/32')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--text-batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--domain-prompt', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_root = os.path.join(args.data_root, 'train')
    image_folder = ImageFolder(train_root)
    model, preprocess, tokenize = load_clip_backend(args.clip_model, device)
    dataset = ImagePathDataset(image_folder.samples, preprocess)

    class_texts = []
    for class_name in image_folder.classes:
        normalized_name = class_name.replace('_', ' ')
        class_texts.extend([prompt.format(class_name=normalized_name) for prompt in DEFAULT_CLASS_PROMPTS])

    flat_text_features = encode_texts(model, tokenize, class_texts, device, args.text_batch_size)
    class_text_features = flat_text_features.reshape(len(image_folder.classes), len(DEFAULT_CLASS_PROMPTS), -1).mean(axis=1)
    class_text_features = class_text_features / np.linalg.norm(class_text_features, axis=1, keepdims=True).clip(min=1e-12)

    domain_prompt = args.domain_prompt or DOMAIN_PROMPTS[args.dataset]
    domain_features = encode_texts(model, tokenize, [domain_prompt], device, args.text_batch_size)
    out_domain_features = encode_texts(model, tokenize, DEFAULT_OUT_DOMAIN_PROMPTS, device, args.text_batch_size)
    image_features, bad_records = encode_images(model, dataset, device, args.batch_size, args.num_workers)

    np.save(os.path.join(args.output_dir, 'clip_image_features.npy'), image_features.astype(np.float32))
    np.save(os.path.join(args.output_dir, 'clip_text_features.npy'), class_text_features.astype(np.float32))
    np.save(os.path.join(args.output_dir, 'clip_domain_features.npy'), domain_features.astype(np.float32))
    np.save(os.path.join(args.output_dir, 'clip_out_domain_features.npy'), out_domain_features.astype(np.float32))
    write_samples_csv(os.path.join(args.output_dir, 'train_samples.csv'), image_folder.samples, image_folder.classes)
    if len(bad_records) > 0:
        with open(os.path.join(args.output_dir, 'bad_images.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'image_path', 'label', 'class_name'])
            for index, image_path, label in bad_records:
                writer.writerow([index, image_path, label, image_folder.classes[label]])

    print('[Data]')
    print(f'num_samples = {len(image_folder.samples)}')
    print(f'num_classes = {len(image_folder.classes)}')
    print(f'dataset = {args.dataset}')
    print(f'clip_model = {args.clip_model}')
    print(f'output_dir = {args.output_dir}')
    if len(bad_records) > 0:
        print(f'[Warning] bad_images = {len(bad_records)}; details saved to bad_images.csv')


if __name__ == '__main__':
    main()
