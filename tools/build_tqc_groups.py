# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from torchvision.datasets import ImageFolder


GROUP_TO_TRAIN_ID = {
    'out_domain': 0,
    'family_mismatch': 0,
    'fine_uncertain': 0,
    'anchor': 1,
    'sibling_boundary': 2,
}


def normalize(features):
    return features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)


def percentile(values, q):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, q))


def softmax(values):
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum().clip(min=1e-12)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def log_line(lines, text):
    print(text)
    lines.append(text)


def write_log(output_dir, lines):
    with open(os.path.join(output_dir, 'tqc_build.log'), 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def quantile_summary(values):
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'p10': percentile(values, 10),
        'p20': percentile(values, 20),
        'p50': percentile(values, 50),
        'p70': percentile(values, 70),
        'p90': percentile(values, 90),
    }


def plot_group_counts(output_dir, class_stats, classes):
    group_names = ['anchor', 'sibling_boundary', 'out_domain', 'family_mismatch', 'fine_uncertain']
    x = np.arange(len(classes))
    bottoms = np.zeros(len(classes))
    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.08), 6), tight_layout=True)
    for name in group_names:
        values = np.array([class_stats[i][name] for i in range(len(classes))])
        ax.bar(x, values, bottom=bottoms, label=name, width=0.9)
        bottoms += values
    ax.set_title('TQC group counts per class')
    ax.set_xlabel('class_id')
    ax.set_ylabel('count')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'class_group_counts.png'), dpi=200)
    plt.close(fig)


def plot_anchor_counts(output_dir, class_stats, classes):
    values = [class_stats[i]['anchor'] for i in range(len(classes))]
    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.08), 5), tight_layout=True)
    ax.bar(np.arange(len(classes)), values, width=0.9)
    ax.set_title('Anchor count per class')
    ax.set_xlabel('class_id')
    ax.set_ylabel('anchor count')
    fig.savefig(os.path.join(output_dir, 'class_anchor_counts.png'), dpi=200)
    plt.close(fig)


def plot_margin_histograms(output_dir, margins):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
    for ax, name, values in zip(axes, ['m_domain', 'm_family', 'm_fine'], margins):
        ax.hist(values, bins=60)
        ax.set_title(name)
    fig.savefig(os.path.join(output_dir, 'margin_distributions.png'), dpi=200)
    plt.close(fig)


def make_sample_grid(output_dir, group_name, samples, records, classes, max_images=25, seed=0):
    indices = [r['index'] for r in records if r['group'] == group_name]
    if len(indices) == 0:
        return
    random.Random(seed).shuffle(indices)
    indices = indices[:max_images]
    thumb = 160
    label_h = 54
    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    canvas = Image.new('RGB', (cols * thumb, rows * (thumb + label_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    for pos, idx in enumerate(indices):
        row = pos // cols
        col = pos % cols
        path, label = samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            image.thumbnail((thumb, thumb))
        except Exception:
            image = Image.new('RGB', (thumb, thumb), 'gray')
        x = col * thumb
        y = row * (thumb + label_h)
        canvas.paste(image, (x, y))
        rec = records[idx]
        text = f"id={idx} y={label}\n{classes[label]}\nmd={rec['m_domain']:.3f} mf={rec['m_family']:.3f} mc={rec['m_fine']:.3f}"
        draw.text((x + 2, y + thumb + 2), text, fill='black')
    canvas.save(os.path.join(output_dir, f'samples_{group_name}.jpg'), quality=90)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='Dataset root, e.g. ../datasets/web-bird')
    parser.add_argument('--dataset', required=True, choices=['web-bird', 'web-car', 'web-aircraft'])
    parser.add_argument('--feature-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--K-s', type=int, default=5)
    parser.add_argument('--K-f', type=int, default=10)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--domain-bottom', type=float, default=10.0)
    parser.add_argument('--family-bottom', type=float, default=20.0)
    parser.add_argument('--fine-high', type=float, default=70.0)
    parser.add_argument('--fine-low', type=float, default=20.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    log_lines = []

    image_folder = ImageFolder(os.path.join(args.data_root, 'train'))
    samples = image_folder.samples
    labels = np.array([label for _, label in samples], dtype=np.int64)
    classes = image_folder.classes
    num_samples = len(samples)
    num_classes = len(classes)

    image_features = normalize(np.load(os.path.join(args.feature_dir, 'clip_image_features.npy')).astype(np.float32))
    text_features = normalize(np.load(os.path.join(args.feature_dir, 'clip_text_features.npy')).astype(np.float32))
    domain_features = normalize(np.load(os.path.join(args.feature_dir, 'clip_domain_features.npy')).astype(np.float32))
    out_domain_features = normalize(np.load(os.path.join(args.feature_dir, 'clip_out_domain_features.npy')).astype(np.float32))

    assert image_features.shape[0] == num_samples, 'clip_image_features.npy length does not match train samples'
    assert text_features.shape[0] == num_classes, 'clip_text_features.npy length does not match classes'

    log_line(log_lines, '[Data]')
    log_line(log_lines, f'num_samples = {num_samples}')
    log_line(log_lines, f'num_classes = {num_classes}')
    log_line(log_lines, f'dataset = {args.dataset}')
    log_line(log_lines, '')
    log_line(log_lines, '[TQC Config]')
    log_line(log_lines, f'K_s = {args.K_s}')
    log_line(log_lines, f'K_f = {args.K_f}')
    log_line(log_lines, f'r = {args.r}')
    log_line(log_lines, f'domain_bottom = {args.domain_bottom}%')
    log_line(log_lines, f'family_bottom = {args.family_bottom}%')
    log_line(log_lines, f'fine_high = top {100 - args.fine_high:.0f}%')
    log_line(log_lines, f'fine_low = bottom {args.fine_low:.0f}%')
    log_line(log_lines, f'temperature = {args.temperature}')

    semantic_similarity = text_features @ text_features.T
    sibling_dict = {}
    family_dict = {}
    for cls_id in range(num_classes):
        order = np.argsort(-semantic_similarity[cls_id])
        order = order[order != cls_id]
        sibling_dict[cls_id] = order[:args.K_s].astype(int).tolist()
        family_dict[cls_id] = [cls_id] + order[:args.K_f].astype(int).tolist()

    sim = image_features @ text_features.T
    domain_sim = (image_features @ domain_features.T).reshape(-1)
    out_sim = image_features @ out_domain_features.T
    m_domain = domain_sim - np.max(out_sim, axis=1)

    m_family = np.zeros(num_samples, dtype=np.float32)
    m_fine = np.zeros(num_samples, dtype=np.float32)
    all_classes = np.arange(num_classes)
    for idx, label in enumerate(labels):
        family = np.array(family_dict[int(label)], dtype=np.int64)
        outside = np.setdiff1d(all_classes, family, assume_unique=False)
        in_scores = np.sort(sim[idx, family])[-min(args.r, len(family)):]
        out_scores = np.sort(sim[idx, outside])[-min(args.r, len(outside)):]
        m_family[idx] = float(in_scores.mean() - out_scores.mean())
        sibling = np.array(sibling_dict[int(label)], dtype=np.int64)
        m_fine[idx] = float(sim[idx, label] - np.max(sim[idx, sibling]))

    tau_domain = percentile(m_domain, args.domain_bottom)
    tau_family = np.zeros(num_classes, dtype=np.float32)
    tau_fine_high = np.zeros(num_classes, dtype=np.float32)
    tau_fine_low = np.zeros(num_classes, dtype=np.float32)
    for cls_id in range(num_classes):
        cls_mask = labels == cls_id
        tau_family[cls_id] = percentile(m_family[cls_mask], args.family_bottom)
        tau_fine_high[cls_id] = percentile(m_fine[cls_mask], args.fine_high)
        tau_fine_low[cls_id] = percentile(m_fine[cls_mask], args.fine_low)

    group_names = []
    records = []
    soft_labels = np.zeros((num_samples, num_classes), dtype=np.float32)
    for idx, label in enumerate(labels):
        label = int(label)
        if m_domain[idx] < tau_domain:
            group = 'out_domain'
        elif m_family[idx] < tau_family[label]:
            group = 'family_mismatch'
        elif m_fine[idx] >= tau_fine_high[label]:
            group = 'anchor'
        elif m_fine[idx] >= tau_fine_low[label]:
            group = 'sibling_boundary'
        else:
            group = 'fine_uncertain'
        group_names.append(group)

        if group == 'sibling_boundary':
            candidates = [label] + sibling_dict[label]
            probs = softmax(sim[idx, candidates] / args.temperature)
            soft_labels[idx, candidates] = probs.astype(np.float32)

        records.append({
            'index': idx,
            'image_path': samples[idx][0],
            'label': label,
            'class_name': classes[label],
            'group': group,
            'train_group_id': GROUP_TO_TRAIN_ID[group],
            'm_domain': float(m_domain[idx]),
            'm_family': float(m_family[idx]),
            'm_fine': float(m_fine[idx]),
        })

    counts = Counter(group_names)
    ignored_total = counts['out_domain'] + counts['family_mismatch'] + counts['fine_uncertain']
    used_total = counts['anchor'] + counts['sibling_boundary']
    log_line(log_lines, '')
    log_line(log_lines, '[Group Statistics]')
    for name in ['anchor', 'sibling_boundary', 'out_domain', 'family_mismatch', 'fine_uncertain']:
        log_line(log_lines, f'{name}: {counts[name]} / {num_samples} = {counts[name] * 100.0 / max(num_samples, 1):.2f}%')
    log_line(log_lines, f'ignored_total: {ignored_total} / {num_samples} = {ignored_total * 100.0 / max(num_samples, 1):.2f}%')
    log_line(log_lines, f'used_total: {used_total} / {num_samples} = {used_total * 100.0 / max(num_samples, 1):.2f}%')

    class_stats = defaultdict(lambda: Counter())
    for record in records:
        class_stats[record['label']]['total'] += 1
        class_stats[record['label']][record['group']] += 1

    zero_anchor = []
    low_anchor = []
    class_stats_path = os.path.join(args.output_dir, 'class_group_stats.csv')
    with open(class_stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'total', 'anchor', 'sibling_boundary',
                         'out_domain', 'family_mismatch', 'fine_uncertain', 'used_ratio'])
        for cls_id in range(num_classes):
            total = class_stats[cls_id]['total']
            anchor = class_stats[cls_id]['anchor']
            sibling = class_stats[cls_id]['sibling_boundary']
            used_ratio = (anchor + sibling) * 100.0 / max(total, 1)
            writer.writerow([cls_id, classes[cls_id], total, anchor, sibling,
                             class_stats[cls_id]['out_domain'], class_stats[cls_id]['family_mismatch'],
                             class_stats[cls_id]['fine_uncertain'], f'{used_ratio:.4f}'])
            if anchor == 0:
                zero_anchor.append((cls_id, classes[cls_id], total))
            elif anchor * 100.0 / max(total, 1) < 5.0:
                low_anchor.append((cls_id, classes[cls_id], total, anchor))

    if counts['anchor'] * 100.0 / max(num_samples, 1) < 15.0:
        log_line(log_lines, f'[Warning] anchor ratio < 15%. Current anchor ratio = {counts["anchor"] * 100.0 / max(num_samples, 1):.2f}%.')
    if len(zero_anchor) > 0:
        log_line(log_lines, f'[Warning] {len(zero_anchor)} classes have zero anchor samples.')
        log_line(log_lines, '[Warning] Classes with no anchor samples:')
        for cls_id, class_name, total in zero_anchor:
            log_line(log_lines, f'class_id={cls_id}, class_name={class_name}, total={total}')
    if len(low_anchor) > 0:
        log_line(log_lines, '[Warning] Classes with anchor ratio < 5%:')
        for cls_id, class_name, total, anchor in low_anchor:
            log_line(log_lines, f'class_id={cls_id}, class_name={class_name}, total={total}, anchor={anchor}')
    if used_total * 100.0 / max(num_samples, 1) < 50.0:
        log_line(log_lines, f'[Warning] effective training samples < 50%. Current effective ratio = {used_total * 100.0 / max(num_samples, 1):.2f}%.')

    log_line(log_lines, '')
    log_line(log_lines, '[Margin Statistics]')
    for name, values in [('m_domain', m_domain), ('m_family', m_family), ('m_fine', m_fine)]:
        stats = quantile_summary(values)
        log_line(log_lines, f'{name}:')
        log_line(log_lines, f'  mean={stats["mean"]:.3f}, std={stats["std"]:.3f}')
        log_line(log_lines, f'  p10={stats["p10"]:.3f}, p20={stats["p20"]:.3f}, p50={stats["p50"]:.3f}, '
                            f'p70={stats["p70"]:.3f}, p90={stats["p90"]:.3f}')

    sb_indices = np.array([r['index'] for r in records if r['group'] == 'sibling_boundary'], dtype=np.int64)
    if len(sb_indices) > 0:
        sb_soft = soft_labels[sb_indices]
        max_prob = sb_soft.max(axis=1)
        entropy = -np.sum(np.where(sb_soft > 0, sb_soft * np.log(sb_soft.clip(min=1e-12)), 0.0), axis=1)
        web_prob = sb_soft[np.arange(len(sb_indices)), labels[sb_indices]]
        top1_is_web = sb_soft.argmax(axis=1) == labels[sb_indices]
        log_line(log_lines, '')
        log_line(log_lines, '[Soft Label Statistics]')
        log_line(log_lines, f'avg_max_prob = {max_prob.mean():.3f}')
        log_line(log_lines, f'avg_entropy = {entropy.mean():.3f}')
        log_line(log_lines, f'avg_web_label_prob = {web_prob.mean():.3f}')
        log_line(log_lines, f'avg_top1_is_web_label = {top1_is_web.mean() * 100.0:.2f}%')
        if max_prob.mean() > 0.95:
            log_line(log_lines, '[Warning] avg soft label entropy too low. T may be too small.')
        if max_prob.mean() < 0.35:
            log_line(log_lines, '[Warning] avg soft label entropy too high. T may be too large.')

    with open(os.path.join(args.output_dir, 'sibling_dict.json'), 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in sibling_dict.items()}, f, indent=2)
    with open(os.path.join(args.output_dir, 'family_dict.json'), 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in family_dict.items()}, f, indent=2)
    with open(os.path.join(args.output_dir, 'sample_group.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': args.dataset,
            'group_to_train_id': GROUP_TO_TRAIN_ID,
            'samples': records,
        }, f, indent=2)
    np.save(os.path.join(args.output_dir, 'soft_labels.npy'), soft_labels)

    with open(os.path.join(args.output_dir, 'sample_margins.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'class_name', 'm_domain', 'm_family', 'm_fine', 'group'])
        for record in records:
            writer.writerow([record['image_path'], record['label'], record['class_name'],
                             f"{record['m_domain']:.8f}", f"{record['m_family']:.8f}",
                             f"{record['m_fine']:.8f}", record['group']])

    plot_group_counts(args.output_dir, class_stats, classes)
    plot_anchor_counts(args.output_dir, class_stats, classes)
    plot_margin_histograms(args.output_dir, [m_domain, m_family, m_fine])
    for group_name in ['anchor', 'sibling_boundary', 'out_domain']:
        make_sample_grid(args.output_dir, group_name, samples, records, classes, seed=args.seed)

    write_log(args.output_dir, log_lines)


if __name__ == '__main__':
    main()
