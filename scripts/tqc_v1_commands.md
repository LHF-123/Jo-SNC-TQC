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

B0 的 `loss_mode` 是 `ce_baseline`，不读取 `sample_group.json` 和 `soft_labels.npy`。它用于和 TQC 分组方法做公平 baseline 对比。

## 4. 跑 B1：Anchor only

```powershell
python main.py --cfg config/bird_tqc_b1.yaml --gpu 0
```

B1 读取：

```text
tqc_group_path: sample_group.json
tqc_sibling_path: sibling_dict.json
```

训练只使用 `group == anchor` 的样本；`sibling_boundary` 和其他组都跳过。

## 5. 跑 B2：Anchor + sibling hard CE

```powershell
python main.py --cfg config/bird_tqc_b2.yaml --gpu 0
```

B2 读取：

```text
tqc_group_path: sample_group.json
tqc_sibling_path: sibling_dict.json
```

训练使用 `anchor` 和 `sibling_boundary`，但两者都用 web label hard CE。

## 6. 跑 B3：Anchor + sibling soft CE

```powershell
python main.py --cfg config/bird_tqc_b3.yaml --gpu 0
```

B3 读取：

```text
tqc_group_path: sample_group.json
tqc_soft_label_path: soft_labels.npy
tqc_sibling_path: sibling_dict.json
```

训练使用 `anchor` 的 hard CE，以及 `sibling_boundary` 的 CLIP soft label CE。

## 7. 跑 B3.1：Anchor + sibling mixed CE

```powershell
python main.py --cfg config/bird_tqc_b31.yaml --gpu 0
```

B3.1 读取：

```text
tqc_group_path: sample_group.json
tqc_soft_label_path: soft_labels.npy
tqc_sibling_path: sibling_dict.json
```

训练使用 `anchor` 的 hard CE，以及 `sibling_boundary` 的 mixed target CE：

```text
q_mix = (1 - soft_label_mix_alpha) * q_clip + soft_label_mix_alpha * onehot(web_label)
soft_label_mix_alpha = 0.5
```

## 8. 跑 B3.2：Anchor + sibling hard CE + soft regularization

```powershell
python main.py --cfg config/bird_tqc_b32.yaml --gpu 0
```

B3.2 读取：

```text
tqc_group_path: sample_group.json
tqc_soft_label_path: soft_labels.npy
tqc_sibling_path: sibling_dict.json
```

训练使用 `anchor` 的 hard CE；`sibling_boundary` 使用 web label hard CE 作为主监督，并加入较弱的 CLIP soft CE：

```text
loss_sb = loss_hard + soft_regularization_mu * loss_soft
soft_regularization_mu = 0.1
```

## 训练时离线文件的使用方式

```text
sample_group.json
  -> data/image_folder.py
  -> batch["group"]
  -> main.py 根据 loss_mode 选择 anchor / sibling_boundary / ignore

soft_labels.npy
  -> data/image_folder.py
  -> batch["soft_label"]
  -> 在 B3 的 sibling soft CE、B3.1 的 sibling mixed CE 和 B3.2 的 soft regularization 中使用

sibling_dict.json
  -> main.py 评估阶段
  -> 计算 sibling_error_ratio

sample_margins.csv / class_group_stats.csv
  -> 复制进实验目录
  -> 仅用于诊断和复盘，不参与训练 loss
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
4. 最后比较 B0 / B1 / B2 / B3 / B3.1 / B3.2 的 test top1/top5。
```
