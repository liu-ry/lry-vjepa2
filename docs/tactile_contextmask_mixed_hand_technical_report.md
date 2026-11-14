# 单手/双手混合训练的两阶段触觉世界模型

## 最新训练结果技术报告

| 版本 | 日期 | 数据 | 训练设备 |
|---|---|---|---|
| v2.0-mixed-hand | 2026-11-11 | single/dual-hand windows + contact gating | 4 × NVIDIA RTX 5090，GPU 2–5 |

> **目标**：在 Stage A 和 Stage B 中同时使用单手、双手触觉窗口。无接触或
> 缺失的一侧通过 `hand_valid` mask 从 attention 和 loss 中排除，而不是复制
> 另一只手或把零值当作真实监督。

---

## TL;DR

- 新 Stage A 最佳 Step **15750**，Step 18750 连续 12 次验证无改善后早停。
- Stage A test：
  - joint relative：**0.1988**
  - future relative：**0.2187**
- 旧双手-only Stage A：
  - joint relative：0.3058
  - future relative：0.3364
- 新 Stage B 最佳 Step **300**，Step 2700 连续 8 次验证无改善后早停。
- Stage B test：
  - tactile relative：**0.1781**
  - event relative：**0.1441**
  - planning relative：**0.1833**
- 旧 Stage B test tactile relative：0.2288，新结果再改善约 **22.2%**。
- 4 个 query token 仍然信息丰富且未塌缩：
  - history/future tactile R²：约 **0.99975**
  - history/future visual R²：**0.9948 / 0.9929**
  - history state R²：**0.9936**
  - mean pairwise cosine：**-0.0380**
  - effective rank：**105.7**
- 需要明确 trade-off：tactile/event/planning relative 明显改善，但 visual
  relative 和 contact/slip probe 略低于旧双手-only 模型。

---

## 1. 单手和双手数据如何定义

真实数据中，“单手任务”不一定缺少另一侧视频文件。部分单手任务仍录制左右
两个 tactile stream，但其中一侧在整个窗口内没有有效 contact activation。
因此不能只按文件是否存在决定有效手。

当前规则为：

```text
hand_valid[h] =
    tactile_stream_exists[h]
    AND
    contact_frames_in_window[h] >= min_hand_contact_frames
```

当前配置：

```yaml
min_hand_contact_frames: 1
encoder_min_contact_frames: 4
min_contact_frames: 34
```

实现含义：

- 真正缺失的 tactile stream 会补零并设置 `hand_valid=False`；
- 文件存在但窗口内没有 contact 的手也会被 mask；
- Stage A 要求窗口内至少有一些有效 contact；
- Stage B 要求 34 帧窗口内至少一只手持续满足 contact gate；
- 左右手分别计算 contact 帧数，不再用简单的左右手 OR 直接当作两只手都有效。

![Mixed-hand validity flow](assets/tactile_contextmask_mixed/hand_mask_flow.png)

---

## 2. Hand mask 如何进入模型

### 2.1 数据 tensor

每条样本统一构造为：

```text
tactile_left  [T,C,H,W]
tactile_right [T,C,H,W]
hand_valid    [2]
contact       [T,2]
slip          [T,2]
```

单手样本固定保持双手 tensor shape，便于 batch 并行。

### 2.2 Stage A

- 无效手的全部 fine encoder tokens 被替换成 mask token；
- coarse pooling 不为无效手产生监督；
- context loss、future delta loss、variance、covariance、temporal loss
  都只在有效 hand 上归一化；
- contact/slip probe 同样只使用有效手。

### 2.3 Stage B

- 无效手 token 不进入 context self-attention 的有效 key；
- future query 的无效手输出强制为零；
- invalid hand 不进入 future tactile loss；
- query condition memory 使用 key padding mask 跳过无效手；
- future visual summary 在有效 hand 上平均；
- current contact/slip probe 只统计有效手。

核心原则：

```text
masked output != zero-valued supervision
```

无效手可以保留固定 tensor 形状，但不会产生梯度或误差。

---

## 3. 实际 hand 比例

本次 held-out test 并非全部为双手：

| Split | Single-hand | Dual-hand |
|---|---:|---:|
| Stage A validation | 50.4% | 49.6% |
| Stage A test | 48.4% | 51.6% |
| Stage B validation | 49.4% | 50.6% |
| Stage B test | 42.5% | 57.5% |

![Hand mix distribution](assets/tactile_contextmask_mixed/hand_mix_distribution.png)

这说明新结果不是只来自双手数据：单手窗口约占 test 的 42%–48%。

---

## 4. Stage A 训练与结果

### 4.1 主 loss

Stage A 仍使用：

```text
L_A =
    L_future
  + 0.1 * L_context
  + 0.05 * L_var
  + 0.01 * L_cov
  + 0.05 * L_contact
  + 0.05 * L_slip
  + 1.0 * L_temporal
```

其中：

- `L_future`：masked future coarse latent delta；
- `L_context`：历史 coarse latent 稳定项；
- `L_var/L_cov`：防塌缩和降冗余；
- `L_contact/L_slip`：detached semantic probes；
- `L_temporal`：触觉特征变化与形变能量一致。

所有 hand 相关项都只在 `hand_valid=True` 的手上归一化。

### 4.2 收敛

- 最大上限：30000 step；
- 至少训练：5000 step；
- best checkpoint：**15750**；
- early stop：**18750**；
- 早停 patience：12 个验证点。

![Stage A convergence](assets/tactile_contextmask_mixed/stage_a_convergence.png)

### 4.3 与旧双手-only 结果对比

| 指标 | 旧双手-only | 新 mixed-hand | 变化 |
|---|---:|---:|---:|
| Test joint relative | 0.3058 | **0.1988** | 改善 35.0% |
| Test future relative | 0.3364 | **0.2187** | 改善 35.0% |
| Test contact AUROC | 0.5807 | 0.5320 | 下降 |
| Test slip AUROC | 0.6734 | 0.6577 | 下降 |

![Stage A old vs new](assets/tactile_contextmask_mixed/stage_a_old_vs_new.png)

结论：

- mixed-hand mask 显著改善未来 latent 预测；
- semantic probe 没有同步提升；
- 当前 Stage A 更偏向稳定的未来增量表示，而不是单纯优化 contact/slip 分类。

---

## 5. Stage B 训练与结果

### 5.1 输入和输出

Stage B 只使用历史：

```text
4 × tactile history
4 × RGB history → 2 V-JEPA temporal steps
4 × state history
```

输出：

```text
future_tactile:   [B,30,2,1,384]
future_visual:    [B,15,D_visual]
condition_tokens: [B,4,384]
```

Stage A encoder 的 `lr_encoder=0`，latent space 保持固定。

### 5.2 主要 loss

```text
L_B =
    L_tactile_delta
  + 1.0 * L_visual
  + 0.5 * L_condition_summary
  + 0.5 * L_future_trajectory
  + 0.05 * L_query_diversity
```

event/planning tactile loss 和 query summary 都只使用有效 hand。

### 5.3 收敛

- 最大上限：30 epochs；
- 至少训练：2000 step；
- best checkpoint：**Step 300**；
- early stop：**Step 2700**；
- 早停 patience：8 个验证点。

![Stage B convergence](assets/tactile_contextmask_mixed/stage_b_convergence.png)

### 5.4 全量 test

Stage B 全量 test：**1630 windows / 815 records**。

| 指标 | Validation | Test | 旧 Test | 结论 |
|---|---:|---:|---:|---|
| tactile relative | **0.1677** | **0.1781** | 0.2288 | 新模型更好 |
| event relative | 0.1399 | **0.1441** | 0.1747 | 新模型更好 |
| planning relative | 0.1720 | **0.1833** | 0.2371 | 新模型更好 |
| visual relative | 0.6953 | 0.7027 | 0.6727 | 新模型较差 |
| contact probe AUROC | 0.6251 | 0.6411 | 0.6696 | 新模型较低 |
| slip probe AUROC | 0.6579 | 0.6951 | 0.7340 | 新模型较低 |

![Stage B old vs new](assets/tactile_contextmask_mixed/stage_b_old_vs_new.png)

相对旧模型：

- tactile relative 改善约 **22.2%**；
- event relative 改善约 **17.5%**；
- planning relative 改善约 **22.7%**；
- visual relative 变差约 **4.5%**；
- semantic probe 下降。

这是当前方案最重要的收益和代价，不能只报告触觉指标。

---

## 6. 4 个 Query Token

Query token 使用 validation 拟合 ridge probe，在 held-out test 评估。

| Target | Test R² | Test cosine |
|---|---:|---:|
| History tactile | 0.99975 | 0.99988 |
| Future tactile | 0.99976 | 0.99988 |
| History visual | 0.99480 | 0.99741 |
| Future visual | 0.99285 | 0.99645 |
| History state | 0.99356 | 0.99756 |

![Query token information](assets/tactile_contextmask_mixed/query_token_information.png)

token 几何：

- mean pairwise cosine：**-0.0380**
- effective rank：**105.7**
- token diversity：**0.9983**

![Query token geometry](assets/tactile_contextmask_mixed/query_token_geometry.png)

结论：

- 4 个 token 同时包含历史触觉、未来触觉、视觉和 state 信息；
- 没有塌缩为同一方向；
- 当前结果证明集合信息丰富，但没有证明 4 个槽形成强逐槽分工。

---

## 7. 为什么使用 mixed-hand mask

不使用以下替代方案：

1. 复制单手到另一只手；
2. 补零后让零值参与 loss；
3. 只用 contact OR 判断两手都有效；
4. 丢弃所有单手窗口。

当前方案：

- 保留真实单手数据；
- 保持 tensor shape 固定；
- 通过 attention/loss mask 排除无效手；
- 仅在有效 hand 上计算依赖触觉的监督。

---

## 8. 结论与后续

### 已达到

- single/dual-hand 同一 batch 混合训练；
- hand mask 贯穿 Stage A encoder、Stage B attention 和全部 hand-dependent loss；
- Stage A/B 都训练到验证平台并由早停结束；
- 未来触觉 latent 显著优于旧双手-only 模型；
- 4 个 query token 保持高可解码性和高秩多样性。

### 仍需改进

1. visual relative 变差，说明需要单独校准视觉 loss 权重或视觉 warmup。
2. contact/slip probe 下降，说明未来触觉精度与局部语义仍有 trade-off。
3. 当前主 Stage B 触觉输出仍是每手一个 384 维 latent，未保留空间网格。
4. 4 个 token 信息丰富，但逐槽角色分工仍不明确。

---

## 9. 复现

### 配置

```text
configs/train_2_1/tactile_contextmask.yaml
```

### Stage A

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 \
  app/vjepa_2_1/train_tactile_two_stage.py \
  --config configs/train_2_1/tactile_contextmask.yaml \
  --stage encoder
```

### Stage B

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 \
  app/vjepa_2_1/train_tactile_two_stage.py \
  --config configs/train_2_1/tactile_contextmask.yaml \
  --stage predictor \
  --encoder-checkpoint \
  data/runs/tactile_contextmask_mixed_20261111/stage_a/best_encoder.pt
```

### 最优权重

```text
data/runs/tactile_contextmask_mixed_20261111/stage_a/best_encoder.pt
data/runs/tactile_contextmask_mixed_20261111/stage_b/best_predictor.pt
```

### 评估与日志

```text
data/runs/tactile_contextmask_mixed_20261111/stage_a/test_metrics.json
data/runs/tactile_contextmask_mixed_20261111/stage_b/test_metrics.json
data/runs/tactile_contextmask_mixed_20261111/stage_b/FINAL_CONDITION_TOKENS.json
data/runs/tactile_contextmask_mixed_20261111/logs/stage_a.log
data/runs/tactile_contextmask_mixed_20261111/logs/stage_b.log
data/runs/tactile_contextmask_mixed_20261111/logs/condition_tokens.log
```

---

## 10. 一句话总结

> 通过 `hand_valid` 将单手、无接触手和双手窗口统一到固定双手 tensor 中，
> 并在 Stage A encoder、Stage B attention、future/query loss 全链路 mask，
> 新模型把 Stage B tactile relative 从 0.2288 降到 0.1781，同时保持 4 个
> query token 的高信息量和非塌缩几何。
