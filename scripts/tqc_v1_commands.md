# TQC V1 启动命令说明

本文件只记录启动命令和执行顺序，不是可直接运行的脚本。建议按步骤逐条复制执行，方便观察每一步日志和中间文件是否正常。

## 默认路径

```text
Dataset: web-bird
Data root: ../datasets/web-bird
TQC output dir: ../results/tqc_web_bird
GPU: 0
```

## 1. 离线提取 CLIP 特征

```powershell
python tools/extract_clip_features.py --data-root ../datasets/web-bird --dataset web-bird --output-dir ../results/tqc_web_bird --gpu 0
```

预期输出：

```text
clip_image_features.npy
clip_text_features.npy
clip_domain_features.npy
clip_out_domain_features.npy
train_samples.csv
```

## 2. 构建 TQC 分组和诊断文件

```powershell
python tools/build_tqc_groups.py --data-root ../datasets/web-bird --dataset web-bird --feature-dir ../results/tqc_web_bird --output-dir ../results/tqc_web_bird
```

重点检查终端输出：

```text
[Group Statistics]
[Margin Statistics]
[Soft Label Statistics]
[Warning]
```

预期输出：

```text
sample_group.json
soft_labels.npy
sample_margins.csv
class_group_stats.csv
sibling_dict.json
family_dict.json
class_group_counts.png
class_anchor_counts.png
margin_distributions.png
samples_anchor.jpg
samples_sibling_boundary.jpg
samples_out_domain.jpg
```

## 3. 跑 B0：Web label CE baseline

```powershell
python main.py --cfg config/bird_tqc_b0.yaml --gpu 0
```

## 4. 跑 B1：Anchor only

```powershell
python main.py --cfg config/bird_tqc_b1.yaml --gpu 0
```

## 5. 跑 B2：Anchor + sibling hard CE

```powershell
python main.py --cfg config/bird_tqc_b2.yaml --gpu 0
```

## 6. 跑 B3：Anchor + sibling soft CE

```powershell
python main.py --cfg config/bird_tqc_b3.yaml --gpu 0
```

## 训练日志重点

每个实验目录需要重点看：

```text
config.yaml
log.txt
msg-log.txt
tqc_epoch_metrics.csv
metrics.json
test_acc.csv
per_class_acc.csv
best_checkpoint.pth
```

判断顺序：

```text
1. 先看 sample_group.json / class_group_stats.csv，确认 anchor 覆盖是否健康。
2. 再看 soft label 统计，确认 avg_max_prob / entropy / web_label_prob 是否合理。
3. 再看 tqc_epoch_metrics.csv，确认 loss_anchor / loss_sb 是否正常下降。
4. 最后比较 B0 / B1 / B2 / B3 的 test top1/top5。
```
