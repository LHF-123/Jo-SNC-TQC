# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : eval.py.py
#   @Author      : Zeren Sun
#   @Created date: 2022/11/18 10:28
#   @Description :
#
# ================================================================
import torch
import os
import csv
from tqdm import tqdm
from utils.utils import AverageMeter
# from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC


def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    """
    Computes the precision@k for the specified values of k in this mini-batch
    :param y_pred   : tensor, shape -> (batch_size, n_classes)
    :param y_actual : tensor, shape -> (batch_size)
    :param topk     : tuple
    :param return_tensor : bool, whether to return a tensor or a scalar
    :return:
        list, each element is a tensor with shape torch.Size([])
    """
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def evaluate(dataloader, model, dev, topk=(1,), progress_bar=False):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :param progress_bar: [bool]   whether or not show progressbar
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_accuracy = AverageMeter()
    test_accuracy.reset()
    topk_accuracy = AverageMeter()
    topk_accuracy.reset()

    with torch.no_grad():
        pbar = tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc=f'EVALUATING') if progress_bar else dataloader
        for _, sample in enumerate(pbar):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
            if len(topk) > 1:
                topk_accuracy.update(acc[1], x.size(0))
    if len(topk) == 1:
        return test_accuracy.avg
    elif len(topk) == 2:
        return test_accuracy.avg, topk_accuracy.avg
    else:
        raise AssertionError(f'topk is set incorrectly (current topk is {topk})')


def evaluate_detailed(dataloader, model, dev, num_classes, class_names=None, sibling_dict=None,
                      output_dir=None, topk=(1, 5), progress_bar=False):
    model.eval()
    maxk = max(topk)
    total = 0
    top_correct = {k: 0.0 for k in topk}
    per_class_total = torch.zeros(num_classes).long()
    per_class_correct = torch.zeros(num_classes).long()
    sibling_error = 0
    total_error = 0

    if sibling_dict is not None:
        sibling_dict = {int(k): set(int(x) for x in v) for k, v in sibling_dict.items()}

    with torch.no_grad():
        pbar = tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='EVALUATING') if progress_bar else dataloader
        for sample in pbar:
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
            _, pred = logits.topk(maxk, 1, True, True)
            pred_t = pred.t()
            correct = pred_t.eq(y.view(1, -1).expand_as(pred_t))
            batch_size = y.size(0)
            total += batch_size
            for k in topk:
                top_correct[k] += correct[:k].contiguous().view(-1).float().sum().item()

            top1_pred = pred[:, 0]
            top1_correct = top1_pred.eq(y)
            for cls_id in y.unique().detach().cpu().tolist():
                cls_mask = y == cls_id
                per_class_total[cls_id] += cls_mask.sum().detach().cpu()
                per_class_correct[cls_id] += top1_correct[cls_mask].sum().detach().cpu()

            if sibling_dict is not None:
                y_cpu = y.detach().cpu().tolist()
                pred_cpu = top1_pred.detach().cpu().tolist()
                for label, pred_label in zip(y_cpu, pred_cpu):
                    if pred_label != label:
                        total_error += 1
                        if pred_label in sibling_dict.get(label, set()):
                            sibling_error += 1

    metrics = {f'top{k}': top_correct[k] * 100.0 / max(total, 1) for k in topk}
    if sibling_dict is not None:
        metrics['sibling_error_ratio'] = sibling_error * 100.0 / max(total_error, 1)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        class_names = class_names or [str(i) for i in range(num_classes)]
        with open(os.path.join(output_dir, 'per_class_acc.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class_id', 'class_name', 'num_test', 'acc'])
            for cls_id in range(num_classes):
                cls_total = int(per_class_total[cls_id].item())
                cls_correct = int(per_class_correct[cls_id].item())
                cls_acc = cls_correct * 100.0 / max(cls_total, 1)
                writer.writerow([cls_id, class_names[cls_id] if cls_id < len(class_names) else cls_id, cls_total, f'{cls_acc:.4f}'])

    return metrics


def detection_evaluate(prediction, ground_truth):
    # prediction and ground_truth are both indicator vectors containing 0 / 1
    precision_func = BinaryPrecision()
    precision = precision_func(prediction, ground_truth)
    recall_func = BinaryRecall()
    recall = recall_func(prediction, ground_truth)
    f1_func = BinaryF1Score()
    f1_score = f1_func(prediction, ground_truth)
    auroc_func = BinaryAUROC()
    auroc = auroc_func(prediction, ground_truth)
    return precision.item(), recall.item(), f1_score.item(), auroc.item()
