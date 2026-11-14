# Stage A / Stage B Loss 详细说明

## 当前最终训练范式

本文只说明当前保留并采用的两阶段触觉世界模型：

- Stage A：细粒度触觉 token mask + context/future latent prediction；
- Stage B：固定 Stage A latent space，训练因果未来预测器和 4 个 query token；
- Stage B 不使用未来 action 或未来 state；
- 两个阶段都不重建原始未来触觉图像。

> **飞书复制说明**：本文所有公式都保存在可选择、可复制的 LaTeX 文本块中。
> 复制全文后，公式会以代码块形式进入飞书；如果需要飞书原生公式，可点击
> 每条公式上方的“复制 LaTeX”，再粘贴到飞书公式编辑器中。

---

# 一、符号约定

| 符号 | 含义 |
|---|---|
| `E_theta` | 在线触觉编码器 |
| `E_xi` | EMA target 触觉编码器，不接收梯度 |
| `z_t` | 第 t 帧的粗空间触觉 latent |
| `b` | 最后一个 context 帧 latent，作为 copy-last 基准 |
| `sg(.)` | stop-gradient |
| `Pool2x2(.)` | 将细粒度编码 token 池化为每手 2×2 粗 latent |
| `normalize(.)` | 对最后一维做 L2 normalization |
| `SmoothL1` | Huber / Smooth L1 loss |

Stage A 使用：

```text
4 个 context 帧 + 30 个 future 帧 = 34 帧
```

每帧每只手的触觉图像为 `336×224`，patch 为 `16×16`：

```text
21×14 = 294 个细粒度 ViT token / hand / frame
588 个细粒度 ViT token / frame（双手）
```

Stage A 的未来预测目标不是 294 个细粒度 token，而是池化后的：

```text
[B, 30, 2 hands, 2×2, 384]
```

也就是每个未来帧、每只手预测 4 个粗空间 latent。

---

# 二、Stage A Loss

## 1. Context latent loss

设：

- `z_hat_context`：latent predictor 输出的 context latent；
- `z_target_context`：EMA target encoder 产生的 context latent。

```latex
L_context =
SmoothL1(
  z_hat_context / 0.01,
  stopgrad(z_target_context) / 0.01;
  beta = 0.2
)
```

作用：

- 预测 4 个历史帧的粗 latent；
- 抑制在线编码器和 target encoder 之间的表示漂移；
- 提供稳定的历史表示；
- 不让未来预测完全依赖当前帧 copy-last。

当前权重：

```yaml
lambda_context_prediction: 0.1
```

因此 context loss 是辅助项，future loss 才是主目标。

---

## 2. Future delta latent loss

Stage A 不直接回归绝对未来 latent，而是预测未来增量。

设：

- `z_hat_h(t)`：预测的未来触觉 latent；
- `z_target_h(t)`：EMA target 的未来触觉 latent；
- `b`：最后一个 context 帧 latent；
- `h`：left/right hand。

```latex
d_hat_h(t) = z_hat_h(t) - b
```

```latex
d_star_h(t) = z_target_h(t) - b
```

这样模型必须学习“触觉未来如何变化”，而不是只复制当前 latent。

### 2.1 Delta Smooth L1

```latex
L_delta_smooth =
SmoothL1(
  d_hat / 0.01,
  d_star / 0.01;
  beta = 0.2
)
```

### 2.2 Delta 方向 loss

除了绝对值，还约束未来变化方向：

```latex
L_delta_direction =
|| normalize(d_hat) - normalize(d_star) ||_2^2
```

当前权重：

```yaml
latent_direction_weight: 0.1
```

于是：

```latex
L_future =
L_delta_smooth
+ 0.1 * L_delta_direction
```

### 2.3 Change weighting

目标触觉变化越大的 hand/spatial cell，权重越高：

```latex
w =
1 + m / mean(m)
```

其中：

```latex
m = sqrt( mean_dim( d_star^2 ) )
```

作用是避免明显变化的位置被大量近似静止的位置淹没。

---

## 3. Variance loss

Variance loss 作用在细粒度触觉编码 token 上，防止 token 维度方差过小和表征塌缩。

```latex
L_var =
mean_j max(0, 0.1 - std_j)
```

当前权重：

```yaml
lambda_variance: 0.05
```

---

## 4. Covariance loss

设 `C` 为 token 特征协方差矩阵，`D` 为特征维度。

```latex
L_cov =
sum_(i != j) C_ij^2 / D
```

作用是降低不同特征维度之间的冗余相关性。

当前权重：

```yaml
lambda_covariance: 0.01
```

---

## 5. Contact / slip probe loss

Contact probe：

```latex
L_contact =
BCEWithLogits(c_hat, c_target)
```

Slip probe：

```latex
L_slip =
BCEWithLogits(s_hat, s_target)
```

当前权重：

```yaml
lambda_contact: 0.05
lambda_slip: 0.05
detach_probe_aux: true
```

由于 `detach_probe_aux=true`：

- probe 参数会通过 BCE 更新；
- 梯度不会回流到触觉编码器；
- contact/slip 主要作为语义诊断，用来检查触觉 latent 是否保留接触和滑移信息。

---

## 6. Temporal change loss

对相邻帧的池化触觉特征计算余弦变化：

```latex
feature_change_t =
1 - cos(f_t, f_(t-1))
```

由触觉形变能量构造目标变化量：

```latex
target_change_t =
0.05 * clamp(pair_energy / contact_threshold, 0, 2)
```

```latex
L_temporal =
SmoothL1(feature_change, stopgrad(target_change))
```

当前权重：

```yaml
lambda_temporal: 1.0
```

该 loss 会反向传播到在线触觉编码器，使触觉 latent 的时间变化与实际触觉形变过程相关。

---

## 7. Stage A 实际总 loss

当前配置关闭了 raw-frame reconstruction 和 reconstruction delta：

```yaml
lambda_reconstruction: 0.0
lambda_delta: 0.0
```

因此当前实际优化目标为：

```latex
L_StageA =
L_future
+ 0.1 * L_context
+ 0.05 * L_var
+ 0.01 * L_cov
+ 0.05 * L_contact
+ 0.05 * L_slip
+ 1.0 * L_temporal
```

各项作用：

| Loss | 作用 |
|---|---|
| Future delta | 学习未来触觉 latent 变化 |
| Context | 稳定历史 latent，抑制表示漂移 |
| Variance | 防止 token 表征塌缩 |
| Covariance | 降低特征维度冗余 |
| Contact/slip probe | 诊断接触和滑移信息，不塑造 encoder |
| Temporal | 让特征变化与实际形变能量对应 |

---

## 8. Stage A checkpoint selection

训练 loss 和验证选择分数不是同一个东西。

Stage A 的验证选择分数为：

```latex
joint_relative =
(
  0.1 * context_relative
  + 1.0 * future_relative
) / 1.1
```

```latex
score_A =
joint_relative
+ 0.001 * (2 - contact_AUROC - slip_AUROC)
```

其中 relative loss 与 copy-last/零增量基线比较：

```latex
future_relative =
(L_future + eps) / (L_copy + eps)
```

`future_relative < 1` 表示优于 copy-last。

---

# 三、Stage B Loss

Stage B 不再使用 Stage A 的 masked context/future loss。

Stage B 输入严格只有历史数据：

```text
4 × touch history
4 × V-JEPA visual history
4 × state history
```

输出：

```text
future_tactile:   [B, 30, 2 hands, 1, 384]
future_visual:    [B, 15, D_visual]
condition_tokens: [B, 4, 384]
```

Stage A 的 encoder 和 EMA target 被保留：

```yaml
lr_encoder: 0.0
fine_tune_encoder: true
```

因此 encoder 虽然仍存在于优化器中，但学习率为 0，实际效果是固定 Stage A latent space。

---

## 9. Event / planning tactile delta loss

未来 horizon 被拆成：

| 区间 | offsets | 数量 |
|---|---|---:|
| event | t+1 到 t+4 | 4 |
| planning | t+5 到 t+30 | 26 |

分别计算：

```latex
d_hat_event = z_hat_event - b
```

```latex
d_star_event = z_target_event - b
```

```latex
d_hat_planning = z_hat_planning - b
```

```latex
d_star_planning = z_target_planning - b
```

每个区间使用：

```latex
L_delta =
SmoothL1(d_hat / 0.001, d_star / 0.001; beta = 0.2)
+ 0.1 * || normalize(d_hat) - normalize(d_star) ||_2^2
```

当前配置：

```yaml
delta_loss_scale: 0.001
delta_direction_weight: 0.1
```

按未来帧数量加权得到总触觉 loss：

```latex
L_tactile =
(4 * L_event + 26 * L_planning) / 30
```

---

## 10. Visual latent loss

Stage B 同时预测 V-JEPA 未来视觉 latent。

绝对视觉项：

```latex
L_visual_abs =
MSE(normalize(z_hat_visual), normalize(z_target_visual))
```

视觉 delta 项：

```latex
L_visual_delta =
SmoothL1(
  (z_hat_visual - b_visual) / visual_delta_rms,
  (z_target_visual - b_visual) / visual_delta_rms;
  beta = 0.2
)
```

总视觉 loss：

```latex
L_visual =
L_visual_abs + 1.0 * L_visual_delta
```

当前权重：

```yaml
lambda_planning_visual: 1.0
visual_delta_weight: 1.0
```

---

## 11. Condition / query token losses

4 个 query token 通过 cross-attention 读取：

- history tactile；
- history visual；
- history state；
- predicted future tactile；
- predicted future visual。

每个 summary loss 都使用归一化后的 MSE：

```latex
L_summary =
MSE(normalize(y_hat), normalize(y_target))
```

### 11.1 History tactile summary

```latex
L_history_tactile =
MSE(normalize(y_hat_history_tactile),
    normalize(y_target_history_tactile))
```

目标是历史触觉序列的四路统计量：

- spatial mean 的时间均值；
- spatial mean 的时间最大值；
- spatial max 的时间均值；
- spatial max 的时间最大值。

### 11.2 Future tactile summary

```latex
L_future_tactile =
MSE(normalize(y_hat_future_tactile),
    normalize(y_target_future_tactile))
```

目标是完整 30 帧未来触觉 latent 的四路统计量。

### 11.3 Full future tactile trajectory

```latex
L_future_trajectory =
MSE(normalize(z_hat_future_trajectory),
    normalize(z_target_future_trajectory))
```

目标是未来 30 帧、双手的完整空间平均 latent trajectory。

该项用于防止 query token 只保留摘要统计量，而丢失未来时间结构。

### 11.4 History / future visual summary

```latex
L_history_visual =
MSE(normalize(y_hat_history_visual),
    normalize(y_target_history_visual))
```

```latex
L_future_visual =
MSE(normalize(y_hat_future_visual),
    normalize(y_target_future_visual))
```

### 11.5 History state summary

```latex
L_history_state =
MSE(normalize(y_hat_history_state),
    normalize(y_target_history_state))
```

### 11.6 Query diversity loss

对 4 个 query token 做 L2 normalization：

```latex
Q_hat = normalize(Q)
```

计算 token 间 cosine Gram 矩阵，并只惩罚非对角项：

```latex
L_query_diversity =
mean_(i != j) (Q_hat_i dot Q_hat_j)^2
```

作用是抑制多个 query token 塌缩成相同方向。

当前权重：

```yaml
lambda_condition: 0.5
lambda_condition_trajectory: 0.5
lambda_condition_diversity: 0.05
```

---

## 12. Stage B 实际总 loss

当前配置关闭以下项：

```yaml
lambda_contact: 0.0
lambda_slip: 0.0
lambda_magnitude: 0.0
lambda_probe_contact: 0.0
lambda_probe_slip: 0.0
```

因此当前实际优化目标为：

```latex
L_condition_summary =
L_history_tactile
+ L_future_tactile
+ L_history_visual
+ L_future_visual
+ L_history_state
```

```latex
L_StageB =
L_tactile
+ 1.0 * L_visual
+ 0.5 * L_condition_summary
+ 0.5 * L_future_trajectory
+ 0.05 * L_query_diversity
```

当前不参与主优化的项：

- future contact/slip BCE；
- current contact/slip probe BCE；
- magnitude loss；
- 单独的 event/planning absolute latent MSE。

这些量仍会计算和记录，用于诊断、AUROC 和 checkpoint selection。

---

## 13. Stage B checkpoint selection

Stage B 的验证选择分数为：

```latex
score_B =
0.75 * tactile_relative
+ 0.10 * L_condition
+ 0.075 * (1 - contact_AUROC)
+ 0.075 * (1 - slip_AUROC)
```

其中：

```latex
L_condition =
L_condition_summary + L_future_trajectory
```

当前视觉 relative loss 只记录，不参与 checkpoint selection。

---

# 四、梯度作用范围

| 参数或模块 | Stage A | Stage B |
|---|---|---|
| 在线触觉编码器 | 更新 | `lr_encoder=0`，不更新 |
| EMA target encoder | 只通过 EMA 更新 | 固定 |
| Stage A latent predictor | 更新 | 作为 Stage B predictor warm start |
| Dense future predictor | 不使用 | 更新 |
| Contact/slip probes | 更新，但不回传 encoder | 只诊断，不参与主 loss |
| Query token 和 readout heads | 不使用 | 更新 |

---

# 五、核心区别

| 项目 | Stage A | Stage B |
|---|---|---|
| 输入 | 4 历史帧 + 30 未来帧的 masked tokens | 只用 4 帧历史 |
| 预测目标 | 30 帧粗触觉 latent | 30 帧触觉 latent + 15 waypoint 视觉 latent |
| 主 loss | future delta + context auxiliary | future tactile delta + visual + query supervision |
| 是否严格因果 | 否，未来部分 token 在 mask 后仍可见 | 是，严格 history-only |
| 是否重建成原始图像 | 否 | 否 |
| Query token | 没有 | 4 个 |

---

# 六、一句话总结

> Stage A 通过局部细粒度 token mask、context 辅助、future delta、temporal
> 和 anti-collapse loss 学习稳定的触觉 latent space。Stage B 固定这个 latent
> space，只使用历史 touch/vision/state 预测未来触觉与视觉 latent，并通过
> summary、完整 trajectory 和 diversity loss 训练 4 个 query token。
