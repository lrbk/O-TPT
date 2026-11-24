# 基于 CLIP Prompt-Learning Paperlist 的 O-TPT 改进方向

> 分析对象：[zhengli97/Awesome-Prompt-Adapter-Learning-for-VLMs-CLIP](https://github.com/zhengli97/Awesome-Prompt-Adapter-Learning-for-VLMs-CLIP)
> 目标：在 **O-TPT (CVPR'25)** 的 "熵 + 正交" 骨架上做**结构性改进**，避免重蹈 PSR/IM 的 self-training 陷阱；每个方向都有明确的 paperlist 引用与理论依据。

---

## 0. 复盘：O-TPT 的三个可攻击面

对着 paperlist 里 general prompt learning + test-time prompt learning 两个大类交叉看，O-TPT 有 3 个显式弱点：

| 弱点 | 表现 | 相关 paper family |
|---|---|---|
| **F1 只约束文本几何** | Vision 分支完全不动 | MaPLe / MMA / PromptAlign（都做多模态 prompt） |
| **F2 忽略类间语义结构** | 所有类被同等对待（正交假设） | HPT / TCP / LLaMP / TAP（都用类先验或 LLM 知识） |
| **F3 单样本独立优化** | 无跨样本信息利用 | TDA / DMN / DPE / Multi-Cache / DMN（都做 test-time cache） |

**F4 隐式**：所有 test-time 方法都在同一个 pseudo-target 上打转。参考 few-shot 侧（KDPL / PromptKD / KDPL）用 **teacher-student distillation** 引入外部信号。

以下 6 个方向按 (a) 对 F1-F4 的覆盖、(b) 与你现有代码的兼容度、(c) 理论强度 排序。

---

## 方向 1：Multi-modal Orthogonality（对应 F1）—— MaPLe/MMA × O-TPT

### 1.1 借鉴来源
- [MaPLe (CVPR'23)](https://github.com/muzairkhattak/multimodal-prompt-learning) — 文本 prompt 与视觉 prompt 通过 coupling function 联动
- [MMA (CVPR'24)](https://github.com/ZjjConan/Multi-Modal-Adapter) — Multi-Modal Adapter，双分支同时插入
- [PromptAlign (NeurIPS'23)](https://github.com/jameelhassan/PromptAlign) — test-time 双分支 distribution alignment

### 1.2 O-TPT 的问题

O-TPT 只对文本 embedding 矩阵 $W \in \mathbb{R}^{C\times d}$ 做正交化。但 CLIP 的分类是 $\langle f_v(x), w_c\rangle$，如果 vision feature $f_v$ 各类的内部 covariance 高度各向异性，即使 $W$ 完美正交，**判决边界仍会歪**。

### 1.3 方法

引入**视觉 prompt** $\mathbf{p}_v$（放在 ViT patch embedding 后），并对视觉类原型 $\mu_c^v = \mathbb{E}_{x\in \text{cache}_c}[f_v(x; \mathbf{p}_v)]$ 也施加正交约束：

$$
\mathcal{L}_{\text{MM-ortho}} = \underbrace{\|W^\top W - I\|_F^2}_{\text{text orthogonality (O-TPT)}} + \lambda_v \underbrace{\|M_v^\top M_v - I\|_F^2}_{\text{visual prototype orthogonality}} + \lambda_c \underbrace{\|W^\top M_v - I\|_F^2}_{\text{cross-modal alignment}}
$$

第三项是 self-duality（NC3 in Neural Collapse），保证 text-vision 分支同构。

### 1.4 与 PSR / NC-TPT 的差别

- 不用 pseudo-target，安全
- 与 NC-TPT 正交（NC 用 simplex ETF；这里可以把 $I$ 替换成 $M^\star$ 得到 **NC-TPT + MM 版本**）
- 与 MMA 差别：MMA 是 few-shot 训练；这里是 test-time zero-shot

### 1.5 实施

在 `otpt_classification.py` 里额外加 `visual_prompt`（learnable token）和 test-time visual prototype cache（每类 top-$k$ 高置信 image feature 平均）。

---

## 方向 2：Cache-Augmented Orthogonality（对应 F3）—— TDA/DMN × O-TPT

### 2.1 借鉴来源
- [TDA (CVPR'24)](https://github.com/kdiAAA/TDA) — training-free dynamic adapter，双 cache（positive/negative）
- [DMN (CVPR'24)](https://github.com/YBZh/DMN) — Dual Memory Networks，静态 + 动态 memory
- [Multi-Cache (ICCV'25)](https://github.com/lyk412/Multi-Cache) — 多缓存提升 test-time generalization

### 2.2 O-TPT 的问题

O-TPT / TPT 都是**逐样本独立**优化，2000 个 test 样本之间的信息完全不共享。而 TDA/DMN 证明 cache 能显著提升 test-time acc。**但 cache 方法不做校准**——TDA 只优化 acc，没有 ECE 保证。

### 2.3 方法：Ortho-Cache TPT

维护一个 test-time cache $\mathcal{C} = \{(f_v(x_i), \hat p_i)\}$，其中 $\hat p_i$ 是前几步的 soft prediction。用 cache 修正 O-TPT 的正交目标：

$$
\mathcal{L}_{\text{ortho-cache}} = \lambda_o \|W^\top W - I\|_F^2 + \lambda_{cc} \sum_c \|W^\top \mu_c^{\text{cache}} - \mathbf{e}_c\|_2^2
$$

第二项让 text embedding $W$ 与 cache 累积的类中心 $\mu_c^{\text{cache}}$ 保持 duality。$\mathbf{e}_c$ 是 one-hot 目标。

### 2.4 关键：cache 里 pseudo label 的处理

**避免 PSR 陷阱**：cache 只存**跨样本聚合**后的 $\mu_c^{\text{cache}}$，且每个 $x_i$ 只在**通过高置信门控**后才入库（$\max_c p(c\mid x_i) > \tau$，$\tau \approx 0.8$）。这样 pseudo label 是**多样本 vote** 而非 view consensus，bias 结构完全不同。

### 2.5 理论支撑

在 online learning 框架下，如果 cache 里正确样本比例 $\ge 1 - \delta$（$\delta$ 依赖于 baseline zero-shot acc），则 $\mu_c^{\text{cache}}$ 是 $\mathbb{E}[f_v \mid y=c]$ 的 $\delta$-perturbed 无偏估计，$\mathcal{L}_{\text{cc}}$ 的极小点使 $W$ 收敛到 NC3 self-duality 点。

### 2.6 与 TDA 的关键差别

TDA 用 cache 直接构造 KNN classifier（加到 logits 上）；我们用 cache **驱动 prompt 优化**（改变 $W$），infra 完全不同，**且 TDA 不做校准我们做**。

---

## 方向 3：LLM-Knowledge-Guided Orthogonality（对应 F2）—— HPT/TCP/LLaMP × O-TPT

### 3.1 借鉴来源
- [HPT (AAAI'24)](https://github.com/Vill-Lab/2024-AAAI-HPT) — Hierarchical Prompt with structured linguistic knowledge
- [TCP (CVPR'24)](https://github.com/htyao89/Textual-based_Class-aware_Prompt_tuning) — Class-aware textual prompt
- [LLaMP (CVPR'24)](https://github.com/zhaohengz/LLaMP) — LLM as prompt learner
- [TAP (ICLR'25)](https://github.com/HHenryD/TAP) — Tree of Attributes prompt learning
- [CuPL (ICCV'23)](https://github.com/sarahpratt/CuPL) — LLM-generated per-class prompt

### 3.2 O-TPT 的问题

$\|W^\top W - I\|_F^2$ 假设**所有类都应该正交**。但真实世界里 "Persian cat" 和 "Ragdoll cat"（Pets 数据集里两类猫）本来就语义相近——强行正交是**违反类间真实相似度**的。

### 3.3 方法：Semantic-Aware Orthogonality

用 LLM（GPT-4 / Llama）生成每类的属性描述，构造**先验类相似度矩阵** $S^{\text{prior}} \in \mathbb{R}^{C\times C}$（余弦相似度 of GPT embedding），把正交目标替换成**保结构目标**：

$$
\mathcal{L}_{\text{sem-ortho}} = \|W^\top W - S^{\text{prior}}\|_F^2
$$

其中 $S^{\text{prior}}_{ij} = \cos(e_{\text{gpt}}(t_i), e_{\text{gpt}}(t_j))$，$t_i$ 是 LLM 生成的类 $i$ 详细描述（e.g., "Persian cat: long-haired, flat face, ..."）。

### 3.4 与 NC-TPT 的关系

- **NC-TPT** 用 simplex ETF（$M^\star_{ij} = -\frac{1}{C-1}$，均等分离）
- **Sem-Ortho** 用 LLM prior（$S^{\text{prior}}$ 允许语义相近类相关）
- 两者互补：**NC 是无先验的最优几何，Sem 是有先验的最优几何**

可以合成 **hybrid**：$\mathcal{L} = \|W^\top W - \alpha M^\star - (1-\alpha) S^{\text{prior}}\|_F^2$，$\alpha$ 控制先验强度。

### 3.5 实施成本

生成 LLM prior 是**一次性 offline 开销**（每个数据集调 GPT-4 一次，$C \le 1000$ 花费 < $10）。test-time 只是查表。**不增加任何在线开销**。

### 3.6 理论支撑

在 KL 散度下，$S^{\text{prior}}$ 是 semantic manifold 的 first-order approximation；O-TPT 的正交约束是 $S^{\text{prior}} = I$ 的特例（假设类完全独立）。**Sem-Ortho 严格更弱且更符合真实结构** → 优化空间更大。

---

## 方向 4：Distribution-Alignment Test-Time（对应 F4）—— PromptAlign × O-TPT

### 4.1 借鉴来源
- [PromptAlign (NeurIPS'23)](https://github.com/jameelhassan/PromptAlign) — 用 source data statistics 做 test-time distribution alignment
- [StatA (CVPR'25)](https://github.com/MaxZanella/StatA) — Realistic test-time adaptation via distribution matching

### 4.2 O-TPT 的问题

O-TPT 只优化当前样本的熵 + 正交，**没利用 ImageNet 训练时的分布先验**（CLIP 训了 400M image-text pair，这些统计信息完全被丢掉）。

### 4.3 方法：Ortho + Token Distribution Alignment

PromptAlign 的核心：让 test-time 的 image feature token statistics（mean, variance across tokens）匹配 source distribution（用 ImageNet 计算的 running mean/var）。加到 O-TPT 上：

$$
\mathcal{L}_{\text{align-ortho}} = \mathcal{L}_{\text{ent}} + \lambda_o \|W^\top W - I\|_F^2 + \lambda_a \sum_l \bigl(\|\mu_l^{\text{test}} - \mu_l^{\text{src}}\|_2 + \|\sigma_l^{\text{test}} - \sigma_l^{\text{src}}\|_2\bigr)
$$

其中 $\mu_l, \sigma_l$ 是第 $l$ 层 ViT token feature 的均值方差。**source stats 是 offline 一次性计算，test 时只做 alignment**。

### 4.4 与 PromptAlign 的差别

PromptAlign 只做 alignment，我们加正交（O-TPT / NC）。**这是纯加法组合**，无冲突。Reviewer 会说 "just combining two methods"，但可以写成**"we show that geometric constraints and distribution alignment are complementary"** 并做消融证明——这本身是有意义的结论。

### 4.5 关键 trick：用 CLIP 自带 stats

不需要重跑 ImageNet，直接用 CLIP 预训练时的 LayerNorm running stats（forward 一次 ImageNet subset 就能得到），**几乎零工程**。

---

## 方向 5：Prompt Distribution Learning（对应 F1+F4）—— ProDA/PLOT × O-TPT

### 5.1 借鉴来源
- [ProDA (CVPR'22)](https://github.com/bbbdylan/proda) — Prompt Distribution Learning
- [PLOT (ICLR'23)](https://github.com/CHENGY12/PLOT) — Prompt Learning with Optimal Transport
- [VaMP (NeurIPS'25)](已列在 paperlist) — Variational Multi-Modal Prompt Learning

### 5.2 O-TPT 的问题

O-TPT 学一个**点估计** prompt $\mathbf{p}^\star$，miscalibration 本质是**忽略了 prompt 的后验方差**（Bayesian view）。

### 5.3 方法：Ortho-Constrained Prompt Distribution

不学单个 prompt，而学一族 $\{\mathbf{p}_m\}_{m=1}^M$（$M \approx 4\sim 8$），每个 prompt 产生一套 $W_m$，最终 logits 是 ensemble：

$$
p(y\mid x) = \frac{1}{M}\sum_m \operatorname{softmax}(f_v(x) W_m^\top)
$$

正交约束改成"**prompt 之间尽可能多样，同一 prompt 内类间尽可能正交**"：

$$
\mathcal{L}_{\text{proda-ortho}} = \mathcal{L}_{\text{ent}} + \lambda_1 \sum_m \|W_m^\top W_m - I\|_F^2 - \lambda_2 \sum_{m\ne m'} \text{sim}(W_m, W_{m'})
$$

第三项负号是**排斥**（让 prompts 相互多样）。

### 5.4 为什么这个能校准

Ensemble 是**已知的最有效校准方法**（Lakshminarayanan et al. NeurIPS'17）。ProDA + O-TPT 组合把 ensemble 优势带进来，且不像 MC-Dropout 那样引入额外噪声。

### 5.5 成本

$M=4$ 时前向 $\times 4$，但可以 batch 化 $\Rightarrow$ 实际墙钟约 $1.5\times$。**对 test-time 场景可接受**。

---

## 方向 6：Adversarial-Robust Orthogonality（外域用）—— R-TPT × O-TPT

### 6.1 借鉴来源
- [R-TPT (CVPR'25)](https://github.com/TomSheng21/R-TPT) — Improving Adversarial Robustness via TPT

### 6.2 O-TPT 的问题

O-TPT 在 clean data 上校准还行，但在 adversarial perturbation（如 ImageNet-A）上性能崩溃。而 R-TPT 证明 adversarial-aware augmentation 能显著提升 OOD 性能。

### 6.3 方法：Ortho + Adversarial View Aggregation

在 K=64 个视图里额外注入 $K_{\text{adv}}=8$ 个 FGSM 一步扰动视图，正交项不变，熵目标改为 **worst-case view entropy**：

$$
\mathcal{L}_{\text{adv-ortho}} = \max_{k\in \mathcal{V}_{\text{adv}}} H(p(y\mid x_k)) + \lambda_o \|W^\top W - I\|_F^2
$$

**关键**：对 clean view 用平均熵，对 adv view 用 max 熵——保证 worst-case robust。

### 6.4 与 NC-TPT 的关系

正交，可叠加。NC-TPT 用 ETF 处理几何，R-TPT 变体处理鲁棒——**联合方法**：NC-TPT + Adv view = "几何最优 + worst-case robust"。这是 CVPR-tier 的完整故事。

---

## 综合推荐 & 优先级

按"**理论强度 × 可行性 × 差异化**"排序：

| Rank | 方向 | 理论强度 | 实现成本 | 与 O-TPT 差异化 | 建议 |
|---|---|---|---|---|---|
| ⭐⭐⭐ | **NC-TPT (方向 A)** | 高（NC theorem） | 低（改 1 行 Gram） | 严格推广 | **已定，先做** |
| ⭐⭐ | **方向 3 Sem-Ortho** | 中（LLM prior） | 中（GPT-4 offline） | 打破正交假设 | **强烈推荐做第二个 method** |
| ⭐⭐ | **方向 2 Ortho-Cache** | 中（online learning） | 高（要写 cache） | 引入跨样本信息 | 与 NC-TPT 组合 |
| ⭐ | **方向 4 PromptAlign+** | 低（组合方法） | 低 | 分布层面 | 消融用 |
| ⭐ | **方向 1 MM-Ortho** | 中（NC3） | 高（改 vision 分支） | 双模态 | 后期扩展 |
| ⭐ | **方向 5 ProDA-Ortho** | 中（ensemble） | 中（$M\times$ forward） | Bayesian view | 附录消融 |
| ⭐ | **方向 6 Adv-Ortho** | 低 | 中 | OOD 专用 | 只在 ImageNet-A 章节 |

## 最优组合方案（如果一次要出一篇顶会）

**主方法**：`NC-TPT (方向 A) + Semantic-Aware Prior (方向 3)`

$$
\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{ent}} + \lambda \bigl\|W^\top W - \bigl(\alpha M^\star_{\text{ETF}} + (1-\alpha) S^{\text{prior}}_{\text{LLM}}\bigr)\bigr\|_F^2
$$

- **NC** 提供无先验最优几何
- **Sem** 提供有先验语义结构
- $\alpha$ 是可学习标量或 $\alpha=0.5$ 固定
- 消融证明单独 NC 也强、单独 Sem 也强、组合最强

**Story**：C-TPT / O-TPT / D-TPT 都在"无先验假设各类独立"，NC 把独立性提升到 simplex ETF 结构，Sem 引入 LLM 语义先验，两者从"无信息 → 结构信息 → 语义信息"三级递进，且都有理论支撑。

---

## 明确不推荐的方向

以下 paperlist 里的方法**不建议**在 TPT 上做：

| 方向 | 为什么不做 |
|---|---|
| PromptKD 蒸馏 | 需要更大 teacher (ViT-L/14)，test-time 场景开销不可接受 |
| DePT 解耦 prompt | Base-to-Novel 场景专属，与 TPT 的 zero-shot 场景不兼容 |
| CasPL 级联 prompt | 需要多阶段训练，test-time 无法级联 |
| SkipT skip tuning | 是 few-shot 优化，与 TPT 独立 |
| DPC dual prompt | 属于 base-to-novel，机制不迁移 |

---

## 与你现有代码栈的对接

`otpt_classification_psr_im.py` 里改动最小的方向排序：

1. **NC-TPT**：改 1 行（`I` → `M_star`），CLI 加 `--lam_etf` — 已在 NC_TPT_method.md
2. **Sem-Ortho**：改 1 行 + offline LLM prior 文件（每数据集一个 pkl），CLI 加 `--sem_prior_path`
3. **Ortho-Cache**：需新增 `TDACache` class（可以直接拷 TDA 官方实现），加 200-300 行
4. **MM-Ortho**：需要改 CLIP forward，插入 visual prompt token，200+ 行改动
5. **ProDA-Ortho**：需要改 prompt learner 为 $M$-copy，模型显存 $\times M$

**建议路线**：先跑 NC-TPT (Phase 1 已规划) → 若成 → 加 Sem-Ortho 做组合表 → 投稿。若 NC-TPT 不成 → 直接跳到 Ortho-Cache（换 F3 战场）。

---

## 一句话总结

O-TPT 有 4 个可攻击面：F1 单模态、F2 无类先验、F3 无跨样本、F4 单目标；paperlist 里对应家族分别是 MaPLe/MMA、HPT/TCP/LLaMP、TDA/DMN、PromptAlign/ProDA。**NC-TPT + Semantic-Aware Prior** 用最小改动同时打 F2 的两个变体（无先验最优几何 + 有先验语义结构），是当前投稿性价比最高的组合。
