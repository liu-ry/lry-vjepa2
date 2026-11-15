# 触觉编码器、未来预测与 horizon 的因果性 autoresearch

日期：2026-11-14

## 结论摘要

这次研究没有否定“旧 h=30 配方产生了当前最佳模型”这一事实，但推翻了把该
结果直接解释成“触觉编码器自身必须编码未来 30 帧”的因果结论。旧对照至少
混入了三个因素：

1. Stage B 的未来 target 被重新编号到时间位置 `0..29`，而预测 query 使用
   `4..33`。
2. h=1/5/10 的 Stage A 只训练 1000 step，h=30 主模型训练到 7750 step。
3. h=1/5/10 的 checkpoint 只有 5/9/14 个时间 embedding；Stage B 将其余
   位置复制为最后一个已训练位置后，又直接用这些位置生成未来 target。

当前证据支持以下新判断：

- `per_frame_causal` 下的编码器输出满足因果性：`h_t` 在运行时只依赖
  `x_0..x_t`，不读取未来输入。
- 使用未来标签反向塑造编码器参数属于预测性表征学习，不等同于未来输入
  泄漏；必须把“运行时信息流”和“训练目标梯度”分开讨论。
- 未来触觉本身通常不能由历史触觉唯一决定，应由 Stage B 的视觉、state 和
  predictor 承担不确定性，而不是要求 Stage A 的触觉编码器预测未来。
- 旧 horizon 结论不稳健。相同 Stage B 500 step 下，h=1 的 test tactile
  relative 为 **0.1860**，h=30 为 **0.1983**，h=1 略优。
- 移除 Stage B 的 visual/state history 后，h=1 test tactile relative 为
  **0.1852**，与原模型 **0.1860** 基本一致；当前 passive 数据上 visual
  history 对触觉 latent 预测没有可测增益。
- 修正“未来 target 重编号”错误后，同一 h=1 checkpoint 在 validation
  step 250 的 tactile relative 从 **0.3111 降到 0.2232**，改善 28.3%。

因此，当前最佳模型仍有价值，但“编码器必须带未来 30 帧预测损失”应降级为
一个与训练预算、时间位置和 target 实现混杂的历史经验，而不是已验证机制。

最终重训后，在严格修正的时间轴、固定 split、3 个 Stage-B seed 上，新模型
的 held-out test tactile relative 为 `0.026212 ± 0.000025`，tactile delta
为 `0.085566 ± 0.000082`，显著优于旧 corrected baseline 的
`0.032958/0.124362`。同 target source 的最终配对实验又确认：runtime
causal encoder 可以不用未来输入，但在当前数据上，future-forecast
regularizer 仍然是更有效的 Stage-A 训练目标。

## 1. 概念边界

### 1.1 编码器输出应包含什么

对因果编码器，正确约束是：

```text
z_t = E_theta(x_0, ..., x_t)
```

`z_t` 可以保留与未来有关的线索，例如当前接触面积、法向力变化率、滑移方向、
历史形变速度；但在给定当前观测时，它不应还携带未来观测本身。一个合理定义是
当前接触状态的充分统计量，而不是未来随机变量。

训练时用未来标签选择哪些当前特征值得保留，属于 predictive representation
learning。它和未来观测进入编码器输入不是同一件事：

| 情形 | 运行时泄漏 | 训练标签塑造编码器 | 判断 |
|---|---:|---:|---|
| Causal encoder + future forecast loss | 否 | 是 | 合法的预测性正则 |
| 未来 raw token 作为可见 anchor | 是 | 是 | MAE 式内插，不是纯预测 |
| 未来 target 用错时间位置 | 无输入泄漏 | 是 | 实验实现错误 |
| history encoder + 独立 predictor | 否 | 否或可选 | 最符合本任务假设 |

### 1.2 触觉预测的不可辨识性

只给定历史触觉，未来接触变化常常是多峰的：机器人下一步动作、物体材质、
视觉遮挡和接触几何都会改变结果。没有动作或视觉条件时，最小化 MSE 容易退化
为条件均值或 copy-last。触觉编码器的职责应是稳定地提取“现在发生了什么”，
而 Stage B 负责把历史触觉、视觉和 state 组合成“未来可能发生什么”。

## 2. 当前实现审计

### 2.1 Stage A

主要路径在 `app/vjepa_2_1/train_tactile_chain_stage_a.py`。

- 触觉 ViT 先逐帧编码，再由 causal temporal adapter 做逐空间 token 的因果
  时间 attention。
- `future_frames` 模式把所有未来帧的细 token 替换成 mask token；online
  encoder 的 future frame 不包含可见未来 raw token。
- `spatial_tokens` 模式在每帧未来触觉中保留约 25% 可见 anchor。这个模式允许
  predictor 读取未来输入，属于 masked interpolation，不能用来证明 future
  forecasting 的因果能力。
- future loss 会通过当前历史 token 和 predictor 更新编码器参数。这是预测性
  表征学习，不是未来 raw input 泄漏。

新增的 `masked_encoding` 模式只重构同一帧被遮码的 EMA latent，不设置未来
loss。为避免 Stage A 总在遮码输入上训练、Stage B 却输入未遮码触觉，默认以
50% 概率训练未遮码 batch。

### 2.2 Stage B

主要路径在 `app/vjepa_2_1/train_tactile_chain_stage_b.py`。

- online 触觉只编码 4 个 context 帧。
- visual 和 state 路径也只读取 context。
- predictor 通过 cross-attention 生成 30 帧 future tactile latent。
- 预测逻辑本身没有访问未来 raw tactile、future visual 或 future state。

修正前，未来 target 单独编码 30 帧时被重编号为位置 `0..29`，而 query 使用
`4..33`。同时 h=1/5/10 checkpoint 的 temporal embedding 长度为 5/9/14，
加载到 Stage B 后剩余位置被复制为最后一个 embedding。修正后，完整 34 帧只
编码一次，context 和 future target 分别取原始 `0..3` 与 `4..33`。

`init_from_latent_predictor: true` 在 chain Stage B 中未被实际使用；该入口
只加载 Stage A encoder/EMA，predictor 仍是随机初始化。旧文档中有关
Stage-A predictor warm start 的描述不能直接套用到当前 chain 脚本。

## 3. 实验矩阵

所有对照使用相同 grouped train/val/test split 和 Stage B seed。核心实验位于：

```text
data/runs/tactile_causal_research_20261114/
```

### 3.1 严格配对：只修正 target 时间轴

编码器均为旧 h=1 Stage A checkpoint，Stage B 均为 250 step、batch 8、
seed 2、相同 loss 和 attention mask。

| target 时间轴 | Val tactile relative | Val visual relative |
|---|---:|---:|
| legacy reindexed `0..29` | 0.3111 | 0.9315 |
| absolute full clip `4..33` | **0.2232** | 0.9321 |

触觉 relative 改善 28.3%，visual 基本不变。这说明时间轴错误确实对旧 horizon
结论产生了实质污染。

### 3.2 旧 horizon checkpoint 的修正版 Stage B

h=1/5/10 checkpoint 训练 1000 step；旧 h=30 checkpoint 训练 7750 step。
Stage B 统一训练到 750 step。

| Stage A checkpoint | Val tactile relative | Test tactile relative | Test visual relative |
|---|---:|---:|---:|
| h=1, 1000 step | 0.2058 | 0.1833 | 0.9057 |
| h=5, 1000 step | 0.2133 | 0.1888 | 0.9084 |
| h=10, 1000 step | 0.1993 | 0.1774 | 0.9073 |
| h=30, 7750 step | **0.0484** | **0.0425** | 0.9080 |

这组结果证明长训 h=30 表征仍有很强优势，但每 1000 step 的 h=30 计算量远大于
h=1，且训练 step 数相差 7.75 倍，不能归因于 horizon 单一变量。

### 3.3 equal-step：1000 Stage A step + 500 Stage B step

两个 forecast 分支除 horizon 外使用相同 Stage A 预算、数据、seed 和 Stage B
预算。

| Stage A | Stage B val | Stage B test tactile relative | Test delta | Test copy |
|---|---:|---:|---:|---:|
| forecast h=1 | 0.2094 | **0.1860** | **0.2534** | 1.3581 |
| forecast h=30 | 0.2244 | 0.1983 | 0.2917 | 1.4666 |

equal-step 下 h=1 略优于 h=30。至少在这个预算内，没有证据支持“必须有 30 帧
未来预测监督”。h=30 的 1000-step Stage A val future relative 约 0.059，但
Stage A 训练后期仍在快速下降，说明它需要更多优化；旧最佳结果更可能来自长训，
而不是 horizon 本身。

### 3.4 完全移除未来预测 loss

`masked_encoding` 不把未来 target 写入 Stage A loss，只做同帧遮码表征学习。

| Stage A 方式 | Stage A test relative | 说明 |
|---|---:|---|
| future forecast h=30 | 0.0565 | 含未来预测标签 |
| masked encoding h=30 | 0.1264 | 无未来 loss，纯特征学习 |
| mixed masked/full h=30 | **0.0681** | 无未来 loss，50% batch 保持未遮码输入 |

Stage A 的 same-frame reconstruction 不能直接与 future delta loss 比较，但它
表明编码器可以收敛，不必靠未来预测项才能获得稳定表征。

在 Stage B 使用 EMA future target 时，纯遮码 checkpoint 出现严重 online/EMA
基准错位；同一 checkpoint 改用冻结 online target 后，初始 tactile relative
从约 34 降到约 0.8。这个结果说明下一步应优先统一 Stage B 的 target latent
空间，而不是继续加 horizon。

最终 500-step Stage B 对照：

| Stage A 表征 | Stage B target encoder | Val relative | Test relative | Test delta | Test copy |
|---|---|---:|---:|---:|---:|
| forecast h=1 | EMA | **0.2094** | **0.1860** | 0.2534 | 1.3581 |
| forecast h=30 | EMA | 0.2244 | 0.1983 | 0.2917 | 1.4666 |
| masked h=30 | EMA | 2.0218 | 1.8658 | 2.7824 | 1.4901 |
| mixed masked/full h=30 | EMA | 1.1038 | 1.0265 | 1.4711 | 1.4323 |
| mixed masked/full h=30 | online | 0.2181 | 0.2083 | **0.2110** | 1.0055 |

纯编码目标在 test relative 上比 forecast-h1 高 0.0223（约 12%），但 Stage B
validation 只差 0.0087（约 4%），且绝对 delta 更低。它不是当前最佳配方，
但已经证明“无未来预测 loss 的触觉编码器”可以支撑因果 Stage B；剩余差距更
应通过稳定 target latent、更多 seed 和视觉条件设计解决，而不是先验地要求
Stage A 预测未来。

### 3.5 visual/state history 消融

使用同一 h=1 Stage A checkpoint、同一 Stage B seed 和 500 step 预算，将
visual/state history 置零，并关闭 visual/condition auxiliary loss。

| Stage B 输入 | Val relative | Test relative | Test delta | Test copy |
|---|---:|---:|---:|---:|
| touch + visual + state | 0.2094 | 0.1860 | 0.2534 | 1.3581 |
| touch only | 0.2094 | **0.1852** | **0.2523** | 1.3581 |

这个结果说明，在当前 passive trajectory 上，未来触觉 latent 可以由触觉时序
连续性很好地拟合，visual/state history 暂时是冗余条件。它不能证明历史触觉
唯一决定未来；当前测试只评价真实发生的单一未来，没有评价动作反事实。

## 4. 指标解释上的风险

不同 Stage A encoder 定义不同 latent 空间，因此 tactile relative 只能做同
latent 内的相对比较。旧 h=30 的 test copy baseline 为 3.77，而 h=1 为 1.36；
copy baseline 本身就相差 2.8 倍。严格跨编码器比较还应增加：

1. 从 future latent 到统一 raw-touch 表征的线性 readout。
2. 对 GT target 解码后的共同物理空间 MSE / cosine。
3. 分 horizon error，而不是只看 30 帧平均。
4. 至少 3 个 seed 的均值和方差。

当前脚本已经同时记录 absolute delta 和 copy baseline，但旧报告主要强调
relative，容易掩盖 latent 几何差异。

## 5. 当前建议

优先级从高到低：

1. 保留 `absolute_full_clip` 作为 Stage B 默认 target 模式。
2. 不再把 `future_forecast` 作为 Stage A 编码器训练的默认必需项；先训练
   same-frame masked encoder，再让 Stage B 独立承担未来预测。
3. Stage B 若使用 masked encoder，先比较 `target_encoder_source=ema/online`，
   并在 checkpoint 中记录在线/EMA 对齐误差。
4. 使用与动作/视觉相关的条件做未来预测。纯触觉 history-only 预测应明确报告
   其多峰不确定性，不能只优化 copy-last relative。
5. 用 equal-step、equal-FLOP 和至少 3 seed 重跑 horizon 消融；禁止再把
   7750-step h=30 与 1000-step h=1 直接比较。

## 6. 复现入口

Stage A 无未来预测：

```bash
python app/vjepa_2_1/train_tactile_chain_stage_a.py \
  --config configs/train_2_1/tactile_chain_stage_a.yaml \
  --objective masked_encoding \
  --horizon 30
```

Stage B 绝对时间 target：

```bash
python app/vjepa_2_1/train_tactile_chain_stage_b.py \
  --config configs/train_2_1/tactile_chain_stage_b.yaml \
  --target-future-mode absolute_full_clip \
  --target-encoder-source online
```

严格旧时间轴对照：

```text
--target-future-mode reindexed_frames
```

高并发数据加载若出现 `/dev/shm` 错误，使用：

```text
--sharing-strategy file_system
```

## 7. 2026-11-15 最终修复与重训

本轮把数据划分和模型 seed 解耦为：

```yaml
split_seed: 0
seed: <model seed>
```

因此多 seed 训练共享同一个 train/val/test partition。Stage B checkpoint
选择改为 `selection_mode: tactile`，只以 tactile relative 和轻量 semantic
probe 决定 best，不允许 visual/query auxiliary loss 覆盖触觉目标。

### 7.1 最终 Stage A 搜索

比较项包括：

| Stage A objective | Step budget | Seeds | 结果 |
|---|---:|---|---|
| future forecast h=30 | 8000 | 0/1 | seed1 最优 |
| masked encoding h=30, p=0.5 | 1000 | 0/1/2 | Stage B 明显落后 |
| masked encoding h=30, p=0.25 | 1000 | 0/1/2 | 表征改善但仍不足以替代 forecast |
| masked encoding h=30, p=0.75 | 1000 | 1 | 明显较差 |

最终 Stage A 采用 future forecast h=30、causal temporal attention、
`future_frames` mask、7×7 target grid、8000 step。seed1 的 validation
best 为 step 8000：

```text
joint_relative = 0.02784
future_relative = 0.03059
test joint_relative = 0.02647
test future_relative = 0.02908
```

运行时仍是严格的 causal encoder；future label 只用于选择有用的当前历史
特征，不作为编码器输入。Stage B 依然只接收前 4 帧。

### 7.2 最终 Stage B 配方

- 绝对时间 target：`target_future_mode: absolute_full_clip`
- target encoder：EMA
- predictor：3 层、384 hidden、6 heads
- projection：MLP
- visual/state history：启用
- tactile/visual/condition auxiliary losses：启用
- checkpoint selection：tactile
- 最大训练：4000 step，最少 3000 step，patience 8
- 输出：future tactile latent 和 visual latent；不输出 action

架构消融中 depth4 和 normed-MLP 在 250-step validation 均落后于标准
depth3 MLP，因此最终保留 depth3 + MLP。

### 7.3 最终重训结果

三个正式 Stage-B seed 使用完全相同的固定 split。split 文件 SHA256 一致：

```text
e903baca5893c3ee6b9d73de9d31fb1771b1ad832d331e0585f529e48b1cb2ea
```

数据规模为 6212 条记录、4607/948/657 条 train/val/test 记录，对应
9214/1896/1314 个 train/val/test windows。训练时以 64 个 validation
batch 做连续 checkpoint 选择；最终候选另做完整的 1896-window validation。

| Stage-B seed | Best step | 64-batch val | Full val | Held-out test relative | Test delta | Test visual |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4000 | 0.029371 | 0.025338 | 0.026219 | 0.085590 | 0.709826 |
| 1 | 4000 | 0.029419 | **0.025323** | **0.026178** | **0.085456** | **0.703673** |
| 2 | 4000 | 0.029399 | 0.025351 | 0.026238 | 0.085651 | 0.707806 |

三个 seed 的 held-out test 统计：

| Metric | Mean | Population std |
|---|---:|---:|
| Tactile relative | 0.026212 | 0.0000249 |
| Tactile delta | 0.085566 | 0.0000815 |
| Visual relative | 0.707101 | 0.002561 |
| Contact AUROC | 0.537252 | 0.002216 |
| Slip AUROC | 0.498247 | 0.003835 |

完整 validation 的 tactile relative 均值为 0.025337。训练日志中的
64-batch validation 明显高于完整 validation（约差 0.004），因此最终
不能直接用训练日志中的 `val tactile_rel` 做跨模型绝对比较。

相对旧 corrected h=30 baseline：

| Metric | Old | Final seed1 | Relative improvement |
|---|---:|---:|---:|
| Test tactile relative | 0.032958 | 0.026178 | 20.6% |
| Test tactile delta | 0.124362 | 0.085456 | 31.3% |
| Test visual relative | 0.777013 | 0.703673 | 9.4% |

最终模型位于：

```text
data/runs/tactile_final_autoresearch_20261115/final_model/
```

其中 `model_manifest.json` 记录 checkpoint SHA256、完整指标、seed 统计和
运行接口。Stage-B 只输出 future tactile/visual latent；模型输出字典和参数
中均没有 action，对应的回归测试已经加入
`tests/models/test_tactile_world_model.py`。

### 7.3.1 query-token 信息量

对已训练的 4 个 Stage-B query token 做 held-out ridge 覆盖：

- 每个 token 对全局 future tactile latent 的 R² 约为 `0.8372–0.8374`；
- 四个 token 合起来为 `0.8378`；
- query geometry 的 effective rank 为 `5.35`；
- token 间平均 off-diagonal cosine 为 `0.000416`；
- token standard deviation 为 `1.186`。

因此 query tokens 没有坍缩，且包含较强的触觉状态/预测信息；但单个 token
已经能达到与四个 token 几乎相同的全局覆盖。进一步把未来分为四个时间段，
token_i 对 segment_j 的交叉覆盖矩阵对角均值和非对角均值分别只有
`0.836654/0.836655`。这说明当前 query tokens 是“高信息、低几何相关、
但功能冗余”的全局摘要，而不是四路互相独立的未来信息。

下游使用时建议把它们看成 4 个可用的全局 summary tokens，可以全部拼接，
也可以先用 1–2 个 token 做轻量 probe；不应假设每个 token 已严格对应独立
未来区间。完整结果位于：

```text
final_model/query_token_eval.json
final_model/query_token_segment_coverage.json
```

### 7.4 最终选择与 risk control

- 只按完整 validation 选择 seed1；test 用于确认但不能反向改写 candidate。
- tactile relative 与绝对 delta、visual relative 同步改善，不是 copy baseline
  单独变化造成的假改善。
- 跨 encoder 的 relative 仍是在各自 latent 空间内归一化，不是原始触觉像素
  MSE；因此同时报告 absolute delta、copy baseline 和逐 horizon 曲线，结论
  限定为“该 Stage-B 任务下的表征可用性”，不外推为物理空间唯一排序。
- seed 间 test relative 标准差仅 `2.49e-5`，配方稳定性远高于旧单点结果。
- 对 seed0/1/2 checkpoint 做直接权重平均（model soup）后完整 validation
  tactile relative 为 `0.842057`，因此明确拒绝该方案。
- 曾额外启动 seed3/4，但其早期 seeded runs 没有提供额外机制证据，且会拖慢
  完整 validation；正式最终统计只采用预先设置的 3 个 seed。

最终模型的全量 per-horizon test 显示，触觉 relative 在前五个 horizon
依次为 `0.00951/0.01157/0.01444/0.01603/0.01786`，到 horizon 30 为
`0.03638`，最差点为 horizon 26 的 `0.03885`。这说明提升不是只来自
copy-last 的短期连续项；远期为约 1–4% 的相对误差，但显著高于 1-step。
视觉 relative 在 horizon 2/6/30 分别为
`0.45546/0.66471/0.48202`。完整曲线保存在最终模型的
`per_horizon_test_metrics.json`。

与旧 corrected h=30 baseline 的逐 horizon 对照中，最终模型把前五个
horizon 从 `0.01661/0.01855/0.01880/0.02083/0.02251` 降到
`0.00951/0.01157/0.01444/0.01603/0.01786`，把 horizon 30 从
`0.04872` 降到 `0.03638`。因此改进覆盖整个 30 帧窗口。

### 7.5 same-frame masked encoder 的严格下游对照

等预算 Stage-A 续训中，masked p=0.25 从 step 750 续训到 7750；best
checkpoint 仍然是 step 1000，validation score `0.034636`，后续没有超过。
这说明只把 same-frame masked objective 训练更久，并不能自动补上时序可预测性。

为了排除 Stage-B target-latent 几何的影响，又使用完全相同的 Stage-B
配方、seed 和 `target_encoder_source=online` 做了严格配对：

| Stage-A objective | Step 500 val | Full val | Full test tactile relative | Test delta |
|---|---:|---:|---:|---:|
| forecast h=30 | 0.053833 | **0.049254** | **0.049960** | **0.164232** |
| masked p=0.25, no future loss | 0.196345 | 0.177760 | 0.180959 | 0.198571 |

同一 predictor、target encoder source、loss 和数据下，masked encoder 的
full test relative 是 forecast encoder 的 3.62 倍。这个差距主要来自编码器
训练目标，而不是 EMA/online 错位。masked Stage-B 分支在约 500–750 step
后已明显劣势，因此提前停止；没有继续跑完 4000 step。

严格配对仍是在两个不同 latent 空间内计算 relative。它支持“forecast
encoder 更适合当前 predictor 和 target space”的工程结论，不等同于证明
所有物理触觉误差都按同一比例改善；这也是同时保留绝对 delta 和 per-horizon
指标的原因。

### 7.6 对原假设的最终判断

用户的运行时分界是成立的：

```text
encoder_t = E(observation_0, ..., observation_t)
```

编码器不能读取未来 observation，也不能把未来随机变量作为自己的输入。
当前实现和前缀不变性测试都满足这一点。

但“训练目标中包含未来标签”并不等于“编码信息包含未来”。future forecast
loss 通过当前历史特征反向传播，只是一个 predictive representation
regularizer；它改变的是哪些当前特征被保留，不改变运行时依赖集合。

实验也没有支持“移除 future loss 会得到更好的 Stage-B 输入”。至少在
当前 passive-contact 数据、ViT encoder 和 30-frame predictor 下，forecast
h=30 显著优于等价的 same-frame masked encoder。因此最终方案是：

- 保留 h=30 future forecast 作为 Stage-A 表征学习正则；
- 保证 runtime encoder 严格 causal；
- 把未来预测职责放在无 action 的 Stage-B predictor；
- 用视觉/state history 条件化未来触觉 latent。

这个结论是经验性的训练选择，不是声称触觉编码器能够从历史唯一确定未来。

### 7.7 正式从头重训交付

2026-11-15 又按同一个最终配方从头训练了一套正式交付模型：

```text
data/runs/tactile_formal_final_20261115/final_model/
```

Stage A seed1 best step 8000：

| Metric | Validation | Test |
|---|---:|---:|
| Joint relative | 0.015729 | 0.014086 |
| Future relative | 0.017211 | 0.015414 |

Stage B seed1 best step 3750：

| Metric | Complete validation | Held-out test |
|---|---:|---:|
| Tactile relative | **0.015098** | **0.015516** |
| Tactile delta | 0.106336 | 0.109285 |
| Visual relative | 0.727628 | 0.727151 |
| Contact AUROC | 0.572203 | 0.545059 |
| Slip AUROC | 0.507218 | 0.507569 |

相对之前 final model，Stage-B tactile relative 从 `0.026178` 降到
`0.015516`。但是新 latent 的 copy baseline 也从 `3.264` 变为 `7.044`，
所以不能把 relative 的下降直接解释为 raw-tactile 像素误差等比例下降。

为做共同空间比较，又训练 `predicted future latent -> 7x7 raw tactile
summary` 的 linear readout，在 held-out test 上：

| Model | Raw test MSE | Copy MSE | Raw relative |
|---|---:|---:|---:|
| Previous final | 4.51665e-5 | 2.01997e-5 | 2.2360 |
| Formal new | 4.53282e-5 | 2.01997e-5 | 2.2440 |

两者在 raw space 中只差约 0.36%，新模型略差；因此本次正式交付的选择是
针对“Stage-B 定义下的 latent prediction 和 query-token 可用性”，而不是
声称新模型也提升了 raw tactile reconstruction。需要 raw touch 指标时，应
继续保留上一套模型作为保守备份。

正式模型的 query-token 信息量：

- 4-token future tactile ridge R²：`0.836218`；
- 单 token R²：约 `0.83535–0.83578`；
- effective rank：`5.366`；
- token 间平均 cosine：`0.001144`；
- token std：`1.095`；
- token→未来四段的 diagonal/off-diagonal R²：
  `0.835587/0.835590`。

query token 仍然高信息、未坍缩，但功能冗余，并非四个互补的时间段专家。
下游可全部使用，也可用 1–2 个 token 做轻量 probe。
