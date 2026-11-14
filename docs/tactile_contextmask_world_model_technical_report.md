# 从掩码预训练到稳定触觉未来潜变量

## 两阶段触觉世界模型技术报告

| 版本 | 日期 | 数据划分 | 训练设备 |
|---|---|---|---|
| v1.0 | 2026-11-11 | 严格 episode/group split | 4 × NVIDIA RTX 5090 |

> **目标范围**：给定双手机器人的历史视觉、触觉与本体状态，预测未来
> 30 帧触觉/视觉潜变量，并生成包含历史与未来信息的 4 个 query token。
> 本报告评估的是 **passive latent forecasting**，不包含动作条件反事实预测。
>
> **飞书兼容说明**：报告内所有结果图和架构图均使用 PNG。Markdown 导入时
> 需要同时保留 `docs/assets/tactile_contextmask/`；飞书 HTML 预览和 DOCX
> 已将图片内嵌，可直接复制或导入。

---

## TL;DR

- Stage A 同时使用 **context loss** 与 **future predict loss** 训练触觉编码器。
- 局部 spatial-token mask 优于未来整帧 mask：
  - future latent relative loss：**0.3364 vs 0.3540**
  - joint relative loss：**0.3058 vs 0.3219**
  - contact/slip AUROC：**0.581/0.673 vs 0.535/0.572**
- Stage B 固定 Stage A latent space 后，全量 test tactile relative 达到
  **0.2288**，相对 copy-last 约降低 **77.1%**。
- 未来触觉 latent 从 `t+1` 到 `t+30` 保持稳定：
  - relative loss：**0.147 → 0.240**
  - delta cosine：**0.977 → 0.949**
- 4 个 query token 共同包含丰富的历史和未来信息。独立 ridge probe 的
  history/future tactile R² 均约 **0.99996**。
- 4 个 token 不塌缩，但当前 **没有形成明显的逐槽分工**：单槽已能承载目标，
  leave-one-out 几乎没有性能下降。它们是信息丰富的通用摘要槽。

### 图表索引

| 编号 | 图表 | 章节 | 主要回答的问题 |
|---|---|---|---|
| 图 1 | 两阶段世界模型架构 | §2 | 历史、未来预测和 query readout 如何连接 |
| 图 2 | Mask 粒度机制 | §3.3 | 整帧 mask 与局部 spatial-token mask 有何区别 |
| 图 3 | Stage A mask 对比 | §3.3 | 哪种 mask 的 future 与 joint loss 更优 |
| 图 4 | Stage A 收敛 | §3.4 | 两种 mask 在完整训练过程中的稳定性 |
| 图 5 | Stage B 收敛 | §4.2 | 固定 latent space 后何时达到最佳 |
| 图 6 | Stage B 策略对比 | §4.3 | 随机初始化、warm start 与冻结策略的差异 |
| 图 7 | Stage B 全量测试 | §4.4 | 最终模型在 held-out test 上的完整指标 |
| 图 8 | Horizon 稳定性 | §4.5 | 未来触觉 latent 在 30 帧内是否持续有效 |
| 图 9 | Query token 可解码性 | §5.2 | 4 个 query 是否包含历史与未来信息 |
| 图 10 | Query token 几何 | §5.2 | token 是否塌缩、是否具有高秩多样性 |
| 图 11 | Query token 冗余 | §5.3 | 每个槽是否形成不可替代的分工 |
| 图 12 | Autoresearch 决策路径 | §6.2 | 失败探索如何收敛到最终方案 |

---

## 1. 问题与设计动机

触觉信号具有三个重要特点：

1. 绝对幅值小，不同帧之间变化更小；
2. 接触位置、边缘和滑移等关键信息具有空间结构；
3. 动态过程比单帧语义更重要。

如果直接用普通 MSE 回归未来触觉 latent，模型容易退化为 copy-last：所有未来
时刻都复制当前 token，绝对误差看起来很小，但没有预测未来变化。

因此本文采用以下原则：

- 使用 **latent delta residual prediction**，显式学习未来增量；
- 始终报告 **relative-to-copy-last** 指标；
- Stage A 保持编码器的空间 token 结构；
- Stage B 稳定 Stage A latent space，再训练未来预测器和 query readout。

> 对于相对损失，`1.0` 等价于 copy-last 基线；越低越好。

---

## 2. 整体架构

![Two-stage tactile world-model architecture](assets/tactile_contextmask/architecture_overview.png)

### 2.1 关键配置

| 模块 | 配置 |
|---|---|
| 历史长度 | 4 帧 |
| 未来触觉 | 30 帧 |
| 未来视觉 | 15 个 tubelet waypoint |
| 触觉编码器 | 6-layer ViT，`embed_dim=384` |
| 视觉教师 | 冻结 V-JEPA 2.1 ViT-B/384 |
| 触觉 mask | future spatial tokens，75% |
| Stage A predictor | 2-layer Transformer |
| Stage B predictor | 2-layer dense predictor |
| Query token | 4 |
| 数据划分 | strict grouped train/val/test |

---

## 3. Stage A：Context + Future Prediction

### 3.1 两类监督

Stage A 不再只预测未来，还预测可见历史 token：

$$
\begin{aligned}
L_{\text{stage-A}}
={}& L_{\text{future}}
 + 0.1 L_{\text{context}}
 + 0.05 L_{\text{var}}
 + 0.01 L_{\text{cov}} \\
&+ 1.0 L_{\text{temporal}}
 + 0.05 L_{\text{contact}}
 + 0.05 L_{\text{slip}}
\end{aligned}
$$

- `context loss`：可见历史 token 的预测与 EMA target 一致；
- `predict loss`：未来 token 的 delta 与 EMA target 一致；
- contact/slip probe 只用于诊断，默认 detach，不反向塑造触觉编码器。

### 3.2 为什么 context loss 不是主目标

等权 context + predict 会把 latent 压得过于时间不变：

- context loss 下降；
- copy-last 基线也下降；
- future relative 长期停在 1.0 附近，模型没有学到增量。

将 context 权重降到 `0.1` 后，context 仍能抑制表示漂移，但不会主导编码器。

### 3.3 Mask 粒度消融

比较了两种 **online encoder 输入 mask**：

| 方法 | 角色 | Mask 粒度 | 可见 future token |
|---|---|---|---|
| `future_frames` | 对照 | 未来帧的全部细粒度编码 token | 无 |
| `spatial_tokens` | 最终采用 | 每帧独立随机 mask 75% 细粒度编码 token | 剩余约 25% |

两条分支预测的是同一个完整目标：未来 30 帧、双手各 `2×2` 的 EMA latent。
`future_frames` 表示 **online encoder 完全看不到未来 token**，并不是“只预测
最后一帧”或“重建未来原始图像”。当前配置没有像素 decoder，也没有 raw-frame
reconstruction loss。

这里的 token 粒度需要与预测目标区分：

| 层级 | 单帧单手的空间粒度 | 数量 |
|---|---|---:|
| 触觉 ViT 编码 token | `336×224`，patch `16×16` → `21×14` | 294 |
| 双手编码 token | 2 × `21×14` | 588 |
| Stage A 预测 target | 编码后池化为 `2×2` | 4 |
| Stage A 双手 target | 2 × `2×2` | 8 |
| Stage B 输出 | 空间 mean pooling 为 1 个 latent | 1 |

因此，**2×2 不是触觉编码器的 token 划分**，而是编码后用于未来 latent
监督的粗空间目标。当前 Stage A 实际在编码器内部对每帧 294 个细粒度 token
执行 mask。

![Mask granularity schematic](assets/tactile_contextmask/mask_granularity_schematic.png)

![Stage A mask comparison](assets/tactile_contextmask/stage_a_mask_comparison.png)

| 方法 | Stage A best step | Val joint ↓ | Val future ↓ | Test joint ↓ | Test future ↓ | Test contact/slip AUROC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 整帧 mask | 9000 | 0.3064 | 0.3370 | 0.3219 | 0.3540 | 0.535 / 0.572 |
| 局部 token mask | **8500** | **0.2892** | **0.3180** | **0.3058** | **0.3364** | **0.581 / 0.673** |

**结论**：局部 token mask 同时改善未来预测和语义 probe。

局部 token mask 在 Stage A 是 masked-autoencoding 式预训练；部分未来 token
在预训练阶段可见。该分支不是严格的 history-only future forecaster；真正的
因果预测发生在 Stage B，Stage B 只使用历史 touch/vision/state。

### 3.4 收敛过程

Stage A 不使用固定 1500-step 截断：

- 最大上限：12,000 step；
- 每 250 step 验证；
- 至少训练 3000 step；
- 连续 8 个验证点无改善才早停；
- 局部 token mask 于 step 8500 取得最佳；
- 整帧 mask 于 step 9000 取得最佳。

![Stage A convergence](assets/tactile_contextmask/stage_a_convergence.png)

### 3.5 粗空间 latent 粒度探索

最初尝试 4×4 future latent target grid，预测过于困难，早期 latent relative
长期接近 copy-last。最终使用双手各 **2×2** 粗空间 latent target：

- 保留少量局部空间结构；
- 将每帧单手的未来监督目标从 16 个 grid cell 降到 4 个；
- 在 4 卡训练、固定显存预算下收敛更快。

这解决了 Stage A predictor 的可训练性问题，但仍是明显的空间信息瓶颈。
Stage A 编码器本身保留 294 个细粒度 token；后续如果下游任务需要接触位置或
滑移方向，应优先把预测 target 从 2×2 提升到 4×4、7×7 或更细网格重新做
收敛性消融，而不是误认为编码器只有 4 个 token。

---

## 4. Stage B：稳定 Future Latent + Query Readout

### 4.1 最终配方

Stage B 的关键不是继续大改编码器，而是稳定 Stage A latent space：

1. 从 Stage A encoder 加载 online encoder；
2. 从 Stage A checkpoint 加载 EMA target；
3. 从 Stage A predictor 加载兼容的 context encoder、cross-attention、
   future query 和 future head；
4. Stage B 的 encoder learning rate 设为 `0`；
5. predictor 使用 `5e-5` 学习率；
6. 使用 residual output，初始等价于 copy-last。

探索中发现：如果 Stage B 继续更新 encoder 和 EMA target，验证 loss 会从
约 0.22–0.24 回升到 0.27–0.64。固定 Stage A latent space 后训练更稳定。

### 4.2 收敛与早停

Stage B 上限为 12 epochs：

- best checkpoint：**step 1200**；
- best validation tactile relative：**0.2157**；
- 在 step 2500 连续 6 个验证点无改善后早停。

![Stage B convergence](assets/tactile_contextmask/stage_b_convergence.png)

### 4.3 Stage B 训练策略消融

以下是训练策略探索阶段的验证结果。除最终方案外，其余运行在确认出现退化后
提前终止，因此没有进入最终全量 test 对比。

![Stage B strategy comparison](assets/tactile_contextmask/stage_b_strategy_comparison.png)

| 初始化 | 是否更新 Stage A encoder/target | 最低验证 tactile relative | 后续行为 | 结论 |
|---|---|---:|---|---|
| predictor 随机初始化 | 是 | 0.6391 @ 100 | 400 step 升至 0.8082，高方差 | 放弃 |
| Stage A predictor warm start | 是 | 0.4406 @ 100 | target 漂移，400 step 升至 0.6381 | 放弃 |
| Stage A predictor warm start | 否，`lr_encoder=0` | **0.2157 @ 1200** | 稳定下降并早停 | **采用** |

> 表中使用 validation `tactile_relative`，不是包含多个诊断项的
> `selection_score`。失败分支仅用于同口径的快速验证，不与最终全量 test 混用。

该消融说明两点：

1. Stage A predictor 的时序先验对 Stage B 有显著价值；
2. 固定 Stage A latent space 比同时更新 encoder/target 更稳定。

### 4.4 全量测试结果

Stage B 全量 test：**1630 windows / 815 records**。

| 指标 | Validation | Test | 解释 |
|---|---:|---:|---|
| tactile relative ↓ | **0.2157** | **0.2288** | 约低于 copy-last 77.1% |
| event relative ↓ | 0.1729 | **0.1747** | 近端触觉增量 |
| planning relative ↓ | 0.2223 | **0.2371** | 中远端触觉增量 |
| visual relative ↓ | 0.6646 | **0.6727** | V-JEPA delta |
| contact probe AUROC ↑ | 0.6829 | 0.6696 | 诊断项 |
| slip probe AUROC ↑ | 0.7230 | 0.7340 | 诊断项 |

![Stage B test metrics](assets/tactile_contextmask/stage_b_test_metrics.png)

### 4.5 分 horizon 稳定性

![Horizon stability](assets/tactile_contextmask/horizon_stability.png)

| Horizon | Relative loss ↓ | Delta cosine ↑ |
|---|---:|---:|
| `t+1` | 0.147 | 0.977 |
| `t+5` | 0.175 | 0.971 |
| `t+10` | 0.199 | 0.963 |
| `t+20` | 0.234 | 0.951 |
| `t+30` | 0.240 | 0.949 |

未来越远，误差缓慢上升，但方向余弦仍保持约 0.95。该结果说明模型不是只复制
当前 token，而是在整个 30 帧 horizon 上都学到了有效增量方向。

---

## 5. 4 个 Query Token 的信息含量

### 5.1 Query 能看到什么

4 个 query 共同 attend 以下 memory：

| Memory 类型 | 内容 |
|---|---|
| History touch | 4 帧触觉上下文 |
| History visual | 4 帧 V-JEPA 视觉 latent |
| History state | 4 帧本体状态 |
| Future touch | 预测出的 30 帧触觉特征 |
| Future visual | 预测出的视觉特征 |

每种 memory 加独立 role embedding。Query 输出参与以下监督：

- 历史触觉摘要；
- 未来触觉摘要；
- 完整未来触觉 trajectory；
- 历史/未来视觉摘要；
- 历史 state 摘要；
- query 间余弦多样性。

当前 `lambda_condition_trajectory=0.5`，完整 future trajectory 已参与训练。

### 5.2 独立可解码性

Ridge probe 只在 validation token 上拟合，在 held-out test token 上评估：

![Query token information](assets/tactile_contextmask/query_token_information.png)

| Target | Test R² | Test cosine |
|---|---:|---:|
| History tactile | 0.99996 | 0.99998 |
| Future tactile | 0.99996 | 0.99998 |
| History visual | 0.99587 | 0.99794 |
| Future visual | 0.99383 | 0.99693 |
| History state | 0.99672 | 0.99925 |

四个 token 的平均 pairwise cosine 为 **-0.0326**，样本中心化 effective rank
为 **84.4**。这说明 token 没有塌缩成同一个向量，且集合中保留了丰富信息。

![Query token geometry](assets/tactile_contextmask/query_token_geometry.png)

### 5.3 但 4 个槽并非强分工

逐槽与 leave-one-out probe：

![Query token redundancy](assets/tactile_contextmask/query_token_redundancy.png)

- 每个单独 query 的 future tactile R² 约 **0.977**；
- 4 个 query 合并后约 **0.9787**；
- 去掉任意一个 query 后，变化小于 0.001。

因此当前结论是：

- ✅ 4 个 token 集合富含历史与未来信息；
- ✅ 它们没有塌缩，保持了近正交的槽间几何；
- ❌ 尚未证明 4 个槽各自承担不可替代的角色；
- ❌ 尚未证明 4 个比 2 个或 1 个更适合下游 action expert。

如果需要“角色化 query”，建议显式限制 memory 类型或 horizon，并加入
query dropout、leave-one-query-out 和下游 1/2/4/8-token 消融。

---

## 6. 实验探索过程

### 6.1 实验矩阵

为了避免只展示成功结论，下表将主要分支、观察结果和最终决策放在同一口径下：

| 编号 | 探索分支 | 观察 | 决策 |
|---|---|---|---|
| A0 | 仅未来预测 | 容易依赖近似不变的绝对 latent，无法排除 copy-last | 加入 context 稳定项 |
| A1 | context 与 predict 等权 | context 下降，但 future relative 长期约 1.0 | context 权重降至 `0.1` |
| A2 | 4×4 未来 latent target grid | 目标过细、残差小，收敛困难 | 粗化为双手各 2×2 latent target |
| A3 | 整帧未来 mask | test future relative **0.3540** | 保留为对照 |
| A4 | 局部 spatial-token mask | test future relative **0.3364**，probe 同步改善 | **采用** |
| B1 | predictor 随机初始化并更新 encoder | 最低 validation tactile relative **0.6391**，随后退化 | 放弃 |
| B2 | Stage A warm start 并更新 encoder | 最低 **0.4406**，online encoder 与 EMA target 漂移 | 放弃 |
| B3 | warm start + 固定 Stage A latent space | 最低 **0.2157**，test **0.2288** | **采用** |
| Q1 | 4 个通用 query + 基础摘要监督 | token 信息丰富但槽间相关性偏高 | 增加 diversity 与未来轨迹监督 |
| Q2 | 4-token ridge / leave-one-out 诊断 | 信息含量高、未塌缩，但去槽差异小 | 保留 4 token，并明确“尚未分工” |

失败分支主要用于快速排除方案，因此采用较早的 validation checkpoint 判断退化；
只有 A3/A4 和 B3 进入最终 held-out test。

### 6.2 决策路径

![Autoresearch decision path](assets/tactile_contextmask/exploration_decision_path.png)

### 关键失败与对应修正

| 观察 | 原因 | 修正 |
|---|---|---|
| future relative 长期约 1 | context loss 主导，latent delta 收缩 | context 权重 1.0 → 0.1 |
| 4×4 latent target 很难收敛 | 未来监督目标过多、残差过小 | 粗化为 2×2 latent target |
| Stage B 随机初始化较差 | 失去 Stage A 学到的时序先验 | 兼容 predictor warm start |
| Stage B encoder 微调后指标回升 | online encoder 和 EMA target 漂移 | `lr_encoder=0` |
| query 高度相关 | 4 个通用 query attend 同一 memory | 加入 diversity/trajectory loss |
| test 指标曾不配套 | 只重载 best encoder，未重载 predictor/target | 修复 checkpoint 重载并重评 |

---

## 7. 数据与评估协议

### Stage A

| Split | Records | Windows |
|---|---:|---:|
| Train | 6,989 | 27,956 |
| Validation | 1,421 | 2,842 |
| Test | 995 | 1,990 |

### Stage B

Stage B 只保留完整 `4 + 30` 帧均满足 contact gate 的窗口：

| Split | Records | Windows |
|---|---:|---:|
| Train | 5,640 | 22,560 |
| Validation | 1,163 | 2,326 |
| Test | 815 | 1,630 |

评估使用 grouped split，同一 episode 不跨 train/val/test。

---

## 8. 局限性与适用边界

### 8.1 Stage B 输出仍是空间池化触觉 latent

Stage A 编码器每帧每手输出 21×14=294 个细粒度 token，但未来监督目标被
池化为双手各 2×2。当前 Stage B 又进一步对触觉 token 做 mean pooling，
输出每手一个 latent token：

```text
future_tactile: [B, 30, 2, 1, 384]
```

因此当前最优模型适合：

- 未来触觉状态摘要；
- 风险/事件趋势预测；
- query conditioning；
- 下游 residual correction 的辅助特征。

如果下游必须判断局部接触位置或滑移方向，应把 Stage B 输出扩展为完整 token
grid 或至少 2×2 粗空间 grid。

### 8.2 模型是 passive forecaster

输入只包含历史 touch / vision / state，没有候选未来 action chunk。因此模型
预测的是“过去条件下最可能的未来”，不是“执行某组动作后的反事实未来”。

### 8.3 Query token 信息丰富但不充分分工

高 R² 说明信息存在，不等于每个 token 都有独立语义。当前 4 个槽更适合作为
下游 condition 集合，而不是直接解释为“历史槽、未来槽、视觉槽、状态槽”。

### 8.4 Delta 幅值仍然很小

未来触觉 target delta RMS 约 `0.0023`。虽然相对 copy-last 改善明显，但绝对
变化很小，部署时仍应使用接近真实的触觉反馈和校准阈值。

---

## 9. 复现与产物

### 配置

```text
configs/train_2_1/tactile_contextmask.yaml
```

### 最佳权重

```text
data/runs/tactile_contextmask_best/
  stage_a/best_encoder.pt
  stage_b/best_predictor.pt
```

### 原始评估

```text
data/runs/tactile_contextmask_best/
  evaluation/FORMAL_TEST_TOKEN.json
  evaluation/FORMAL_TEST_FRAME.json
  stage_b/test_metrics.json
  evaluation/FINAL_CONDITION_TOKENS.json
```

### 图表与数据快照

```text
docs/assets/tactile_contextmask/summary.json
docs/assets/tactile_contextmask/*.png
docs/assets/tactile_contextmask/*.svg  # 矢量源文件
app/vjepa_2_1/plot_tactile_contextmask_report.py
app/vjepa_2_1/export_feishu_report.py
```

重新生成图表和飞书版本：

```bash
python app/vjepa_2_1/plot_tactile_contextmask_report.py
python app/vjepa_2_1/export_feishu_report.py
```

---

## 10. 结论

当前最合理且实测最优的范式是：

1. Stage A 使用局部 spatial-token mask；
2. context loss + future delta predict loss 同时监督；
3. context 作为低权重稳定项；
4. 粗空间未来 token 保证可训练性；
5. Stage B 固定 Stage A encoder/EMA target；
6. warm start predictor，避免重新学习时序先验；
7. 用 4 个 query token 汇总历史与预测未来；
8. 用 full future trajectory loss 强化 query 的未来信息。

最终模型已经提供了稳定、明显优于 copy-last 的未来触觉 latent，并生成信息
丰富的 query token。下一步最重要的改进不是继续增加 query 数量，而是：

- 若需要局部触觉几何，恢复 Stage B 的空间 token 输出；
- 若需要动作反事实能力，加入候选 action chunk conditioning；
- 若需要 4 个 query 的可解释分工，增加结构化角色监督和下游 token 数消融。
