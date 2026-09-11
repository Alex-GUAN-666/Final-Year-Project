> **2026-09-11 supplemental review:** The distilled pool is now recovered. Public-tokenizer reconstruction reproduces 452 training examples and shows 5/20 historical reasoning evaluation questions overlap training. Older references below to missing data or unknown overlap are superseded by [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) and [the recovered-data audit](../reports/RECOVERED_DATA_AUDIT.md).

# FYP 两份论文全文审计

审计日期：2026-09-11。审计对象为用户提供的两份论文；本报告不代表已完成 GPU 重训练或重新获得论文分数。原始文件未修改。主要复现目标应为较晚 PDF 中的多教师合成数据与 SFT 流程；较早 DOCX 的 GRPO 和语义路由器应作为独立探索记录。

## 阅读覆盖

| 材料 | 已完成的阅读与核查 |
|---|---|
| Group14_ Research and application of the language models.pdf | 全部 42 页逐页文本，全部 6 张正文表及 3 张附录表，19 个嵌入图像对象（含两张签名图和一个 2×2 像素占位图），16 个编号技术图；另高分辨率查看 p22、23、25、28、31、32、33、35、40–42，以确认公式和表格。 |
| Group14_GRPO-A Resource-Efficient Framework for Structured Mathematical Reasoning and its Role in Hybrid LLM Architectures.docx | 全部 466 个顶层正文块，包括 463 个段落、1 个表格、1 个目录内容控件和节属性；完整阅读附录代码、参考文献、页眉页脚/脚注/尾注 XML 和元数据；渲染为 22 页并检查全部页面缩略图，完整查看 3 个图像和含公式的 p12。 |
| DOCX 文档结构 | 未发现修订插入、修订删除或评论锚点；脚注和尾注部分无实际正文。提取到 4 个原生 OMML 数学对象。 |

PDF 页码指文件中的 1–42 页，也与印刷页码一致。DOCX 页码指本次渲染的 22 页；其旧目录/元数据显示 21 页，跨环境排版可能变动，因此同时给出章节或正文块号。全文提取文件包含私人原始信息，只适合作为审计工作材料，不应直接加入公开仓库。

## 两份论文的关系

| 维度 | 较晚 PDF | 较早 DOCX |
|---|---|---|
| 正文标题 | Enhancing Compact Language Models: Multi-Teacher Distillation and Supervised Fine-Tuning for Research and Application | GRPO: A Resource-Efficient Framework for Structured Mathematical Reasoning and its Role in Hybrid LLM Architectures |
| 作者 | Yuzhen Guan、Weizhou Lu | Guan Yuzhen、Lu Weizhou |
| 时间证据 | 封面 November 2025；声明 2025-11-30；导师/考官页 2025-12-25；PDF 元数据也为 2025-12-25 | 封面 June 2025；文件元数据为 2025-06-27/28 |
| 核心模型 | Qwen3-4B-Base → LoRA SFT → merged 模型 | Qwen3-4B-Base → 结构格式 SFT → GRPO；另有 Qwen3-Embedding-4B 路由器探索 |
| 数据 | 多教师生成的 system/input/output JSONL，CoT 与可执行 Python | OpenMathReasoning-mini 的 59 条结构 SFT 数据；DAPO-Math-17k 子集进行 GRPO；AFQMC/LCQMC 用于嵌入模型 |
| 评价 | 两类各 20 题；执行 Python 后数值匹配 | 训练日志以及少量演示；核心前后对照为 sqrt(101)，无双基准准确率表 |
| GRPO 是否属于最终主线 | p17 明确不使用 GRPO/RL，p18 再次强调仅 SFT，p37 列为未来工作 | 属于核心实验；MoE 集成明确尚未实现（§3，正文块 65–73） |
| 应如何进入 GitHub | 主要可复现流程的论文参考 | 独立 experimental/legacy 说明，避免和 SFT 结果混算 |

“较早探索”和“较晚主线”是由时间及正文内容得出的工作归类，不是对作者当年正式版本决定的额外猜测。两份论文不能拼成“最终完成 SFT+GRPO+MoE 并在双数据集上提高”的单一结论。

## PDF 中可以准确报告的实验结果

来源：p31 §5.1、p32 §5.2、p33 Table 5、p36 §5.5。

| 任务 | 提示策略 | Base | SFT-Merged | 绝对变化 | 增加正确题数 |
|---|---|---:|---:|---:|---:|
| Arithmetic | Code-Only | 14/20 = 70% | 17/20 = 85% | +15 个百分点 | +3 |
| Reasoning / proof-style | CoT + Code | 12/20 = 60% | 15/20 = 75% | +15 个百分点 | +3 |
| Overall | Hybrid routing | 26/40 = 65% | 32/40 = 80% | +15 个百分点 | +6 |

Overall 的 26/40 和 32/40 是由正文分类分子相加推得；Table 5 本身只打印百分数。论文把变化列写作“+15%”，仓库应写“+15 percentage points”，避免和相对提升混淆。对应相对提升分别为 21.43%、25.00%、23.08%，无需同时宣传多套数值。

论文说两个集合各 20 题，并“purposefully kept out of the training data”。它没有附上这 40 道完整题目、稳定样本 ID、每模型逐题输出及全部判分记录。因此这些是“thesis-reported results”，不能标记为本次重跑验证过的结果。用户现在回忆的几百条采样、上传 CSV 的行数以及早期 notebook 的其它试验规模，必须分别登记，不能改变论文表格的原始分母。

评价口径是生成一份 Python 代码、无错误执行、打印数值，并在 δ = 1e-6 容差内匹配；这是使用外部 Python 执行的端到端任务准确率。论文把它称为 Pass@1/Exact Match。它不是纯模型心算准确率，也不是形式化定理证明正确率；代码答案正确本身不证明 CoT 中每步推导正确。

论文没有多随机种子重训、置信区间、配对显著性检验或教师/提示/执行器消融。对 20 题每类增加 3 道的结果，应报告描述性增益，不应宣称已经证明普遍或统计显著提升。p36 自己承认样本量、数据污染风险及硬件限制。

## PDF 方法、提示工程与实现参数

### 数据与教师

- p25 §4.2.1 将提示分为 arithmetic/bitwise/algorithmic 计算题和 geometry/probability/proof 等理论题。
- Table 1 计算提示片段要求定义 def solve()，打印最终答案；理论提示片段要求先逐步解释，再用 fenced Python block 完成计算或检查。完整可运行 prompt 需要以 .py/notebook 原文为准；论文提供的是片段和结构要求。
- p26 §4.2.2 说逐个问题经 Hugging Face transformers 请求教师，检查 system 消息、用户 instruction 和至少一个有效 Python fenced block，再保存 system/input/output JSONL。
- “包含代码块”和“结构有效”不等于数学答案经过验证。真正的运行校验、人工审核比例、失败过滤、去重规则要由生成脚本及数据记录证明。
- 教师名单不统一：摘要/p14 提 Qwen、LLaMA、Mistral；p20/21 举 Gemma-7B 和 Mistral-7B；p26 增加 Phi-3；p11 Figure 3 却举 GPT-4、Qwen2-72B、DeepSeek-R1。正文没有统一列出确切模型 ID、revision、每位教师样本数、生成温度和随机种子。
- p26 Figure 11 写 “via API”，正文写 transformers 接口；需按实际脚本解释调用方式，不能凭概念图断言使用了外部 API。
- p30 表示 70B 教师在硬件约束下发生显存错误。概念图中的大模型例子不能直接作为这些大模型真实成功产出训练数据的证据。

### 训练与合并

| 配置 | 论文证据 | 尚需与代码确认之处 |
|---|---|---|
| Base | Qwen3-4B-Base；p21 截图是 Kaggle 本地 checkpoint 路径 | 精确 revision、文件 hash 和 tokenizer |
| LoRA rank/alpha | Table 2 p28：r=32、alpha=64 | 应从实际运行配置再次核对 |
| 学习率 | 2e-4 | 对应哪一次训练 |
| Global batch size | Table 2：64，经梯度累积实现 | microbatch、累积步数、GPU 数未在表中给全 |
| Max sequence length | 2048 | 超长样本是否过滤/截断、训练 token 数 |
| Optimizer | Table 2 写 AdamW | 8-bit optimizer 与普通 AdamW 区分 |
| Checkpointing / precision | p21 提 fp16、梯度检查点；截图 random_state=3407 | seed 的所有随机源及 deterministic 设置 |
| Adapter targets | p21 截图：q/k/v/o_proj、gate/up/down_proj | p27 正文和图对 down_proj 的处理不同 |
| Merge | p22 Figure 9：model.merge_and_unload()；路径 Qwen3-4B-SFFT-merged | 是否保存了权重与 tokenizer，reload 后输出等价性 |
| 下一 token 目标 | p28：cross-entropy；ChatML | loss masking、packing、epochs/steps、调度器、最终训练日志未完整列在论文 |

最重要的内部矛盾是量化：摘要、p20、p27 和架构图说 4-bit 量化/QLoRA，但 p21 Figure 8 的实际代码截图明确写 load_in_4bit=False、torch_dtype=torch.float16，并带注释“False for LoRA 16bit”。这意味着不能直接从论文宣传“实际训练就是 4-bit QLoRA”。若 notebook 确认为 fp16 LoRA，应如此描述；可以另提供新的 4-bit 配置，但必须标注其为复现改造。

p27 Figure 12 将 down_proj 画为仅冻结、没有 adapter，p21 的实际代码截图却把 down_proj 加入 target_modules。实际 notebook 应优先决定复现参数。

p28 的合并公式 W' = W + (alpha/r)BA 是标准 LoRA 合并说明。论文没有给可下载权重地址、可验证模型版本或模型文件清单。说“导出到 Hugging Face 生态”不能替代真正的已上传模型证据。

### 推理和判分

- p29–30 §4.4.2：select_prompt(problem) 依据 “calculate the value”等关键词路由；计算走 Code-Only，其余走 CoT-and-Code；Figure 14 简写为是否包含 “Calculate”。
- 使用 do_sample=False 的 greedy decoding（p31 Table 3），解析 Python fenced code block，必要时给末尾表达式添加 print，再执行得到数值。
- p30 提正则和 AST 验证，以及避免 SymPy 返回非数值对象的 prompt 调整。论文没有在正文给出完整 allowlist、timeout、文件/网络隔离、执行器进程边界；“AST parser”本身不能证明构成安全沙箱。
- 报告模型、prompt、解析器、代码修复规则和数值容差时应绑定同一个配置版本。任何修复逻辑的变化都有可能改变准确率，需要独立记录。
- p35 的格式失败例子是计算出 42 但未 print；p30 又说自动补 print。这可能是规则覆盖不完整或不同运行版本，应由具体失败样本判断，不能宣称所有此类问题已完全解决。
- 使用关键词路由不是 DOCX 的学习型 semantic embedding router，更不是 MoE 的已实现门控系统。

## PDF 的结论和图表需要修正的地方

1. **SFT 与 logit 蒸馏混写。** p16/18/28 明确纯 SFT、下一 token 交叉熵；但 p22 给 L_KD = τ² KL(p_T^τ || p_S^τ)，p23 给 L_total = λL_KD + (1−λ)L_SFT，p10 Figure 2 也画 soft labels。论文没有给 τ、λ、teacher logits 的采集/存储。若代码只训练教师生成文本，应写“sequence-level / response distillation through SFT”，不声称实现 KL/logit loss。
2. **算术与抽象推理提升强弱缺乏对应证据。** p17/23/37 说计算显著提升而抽象推理只少量提升，但 Table 5 两类完全同为 +15 个百分点。这组表格无法支持两类增益大小不同。
3. **无法归因于多教师本身。** 对照只有 base 与 tuned；没有单教师、纯 CoT、纯代码、不同 prompt 或无执行器对照。可说这个整体 pipeline 的报告分数变好，不能说已量化证明多教师优于单教师。
4. **错误类别没有可核对分母。** p34–35 Figure 16/Table 6 写 runtime/symbolic 45%、logic/heuristic 35%、parsing/format 20%，未给逐例分类、两模型分开计数或采样范围。它们只能作为论文的定性/汇总描述，不能当成恢复出的真实标签。
5. **Figure 15 的纵轴排版不可信。** p33 图中 0、20、40、60、80 的位置大致等距，但顶端 100 紧贴 80，间距不一致。应从 Table 5 的明确数值重画，不从像素高度推导新的准确率。
6. **Figure 5 不宜作为真实输出证据。** p15 图里的乘法输入连接到语法混乱的质数代码；Fibonacci 示例旁出现 Eiffel Tower 文本，输入输出未清楚标为 before/after；它与图注“before and after”不形成可核对实验例子。只能视作不可靠说明图，公开项目用真实保存输出替换。
7. **其它概念图应重绘。** p9 Figure 1 的参数刻度及图例有明显无意义/不一致文本；p16 Figure 6 写 “Cross-Entropy Maximization”但其公式是负 log likelihood，应为最小化；p11 的教师例子与实际名单未对齐；p12 概念图的边方向、占位 “xx%”不构成测量数据。
8. **引用编号错配。** p13 TinyLlama 引 [14]，参考文献 [14] 实为 Visual instruction tuning，[12] 才是 TinyLlama；LLaVA 引 [1] 但 [1] 是 GPT-3；p17 PPO 引 [15] 但 [15] 为 MiniGPT-4、[11] 才为 PPO；Qwen3 的论述引用较早 Qwen/Qwen2 报告。新 README 应重新引用真实对应方法资料，不能复制错配编号。

### 附录的具体覆盖

p40、41、42 各 10 行，共 30 条问题摘要及答案，分 Algebra/Arithmetic、Discrete/Number Theory、Geometry。它没有标明与正式 20+20 基准的交集，也没有逐题模型输出。

附录很多题只有摘要，例如电梯步数、两个矩形、书宽度、三元方程、友好整数等；原始条件缺失时不能据此重新生成“原始测试题”。正文代表题 x+y+z=100 的正整数解数为 4851；附录列的是 x+y+z=10 的非负整数解数 66，二者是不同题。也不能把它们直接当同一样本核对。

原文个别公式的定义域/条件被摘要省略，例如 m²+5n²=36 后求 m+5n 的唯一答案需要额外条件。复现应保存完整 problem 字段，不能以附录摘要替代。

## DOCX 早期 GRPO 研究的完整要点

### 数学专家的训练

- §3.1（p6–7）：第一阶段 59 条 OpenMathReasoning 格式 SFT，2 epochs，learning_rate=2e-4，8-bit Adam；第二阶段 DAPO-Math-17k 的处理后子集，GRPO learning_rate=5e-6。
- §4.1（p8–11，正文块 133–209）：unsloth/Qwen3-4B-Base；max_seq_length=2048；load_in_4bit=False；fast_inference=True；max_lora_rank=32；gpu_memory_utilization=0.7。r=32、alpha=64、七个 attention/MLP projection、Unsloth checkpointing、random_state=3407。
- 报告加载权重 7.63 GiB、加载时间 54.56 秒、KV cache 1.68 GiB（正文块 151）。
- 数据入口：unsloth/OpenMathReasoning-mini，split="cot"，取 expected_answer/problem/generated_solution 字段。
- SFT 示例 per_device_train_batch_size=1、2 epochs、logging_steps=5、optim="adamw_8bit"、weight_decay=0.01、linear scheduler、seed=3407。
- SFT 报告：118 steps，loss 0.645 → 0.1805，0.656 samples/s（正文块 173）。59×2=118 与所给设置相容，但论文只截取部分代码；没有完整展示 Messages 到 text、pandas 到 Dataset 的转换。
- GRPO：max_prompt_length=202、max_completion_length=1846、max_steps=100、num_generations=4；SamplingParams min_p=0.1、top_p=1.0、top_k=-1、seed=3407、以 EOS 停止。
- 四个 reward 函数：match_format_exactly、match_format_approximately、check_answer、check_numbers。论文附录只完整给出其中两个函数和部分正则，不能只复制论文片段就运行完整训练。
- GRPO 报告：最终 loss 0.0001507，reward 从 0.125 到 −0.079，reward std 最大 0.163776，0.055 samples/s，平均 completion 约 200 tokens（§4/5）。更低 loss 或这一 reward 变化本身不等于数学准确率提升，也不能单凭此称“收敛”。
- 推理：temperature=1.0、top_k=50、max_tokens=2048，从 grpo_saved_lora 加载 LoRA。与最终 PDF 的 greedy 执行评测是不同协议。
- 导出示例支持 merged_16bit 和 GGUF q4_k_m；方法调用示例不是可验证模型文件已存在的证据。

### 模板与 reward 的准确内容及缺陷

附录系统提示要求数学专家把推理放入 <start_working_out> 和 <end_working_out>，再把答案放入 <SOLUTION> 和 </SOLUTION>。但附录变量 reasoning_end 实际是 "</end_working_out>"，样例输出正文使用 "<end_working_out>"；模板描述、变量和值之间不完全一致，应采用真实 notebook 中一致的序列。

format_dataset 会去掉 generated_solution 里的 <think>/</think>，将 reasoning 与 expected_answer 包进上述标签，然后返回 system/user/assistant 三条消息。

附录 match_format 正则从 reasoning_end 开始匹配，再找到 SOLUTION，允许尾部 whitespace/EOS；没有强制 response 以 reasoning_start 开头。因此 match_format_exactly 名称比它实际验证的格式更严格。§5 所举单独 "<SOLUTION> 2 </SOLUTION>" 能匹配的示例，与附录要求前置 reasoning_end 的正则不相符。

check_answer 代码的分数是：去空白的字符串完全匹配 +5；数值差小于 1e-6 +3.5；比值绝对值在 0.9–1.1 内 +2；在 0.8–1.2 内 +1.5；其它数值 −2.5；无法转 float −4.5；无法提取答案 −2。§5 文字把 +3.5 说成 space-adjusted match，但代码实际上把 strip 后完全相同给 +5，把数值等价但字符串不同给 +3.5。

近似分数使用 abs(gen_num / exp_num)，会对异号但绝对值接近的答案给正分，例如 true=10、guess=−10 可以得到 +2。这是早期 reward 的实质缺陷，不能作为最终准确率判分器。对 zero 的近似规则也需明确。

### sqrt(101) 示例应如何理解

| 项目 | 论文报告 |
|---|---|
| pre-GRPO | 输出包含 10.0498756 及无关 Wiki 文本；42.55 秒 |
| post-GRPO | 使用规定标签，最终 10.05；92.70 秒 |
| 论文称偏差 | 0.0025 |
| 真正答案绝对误差 | abs(10.05−sqrt(101)) = 0.00012437887911076473 |
| 0.0025 的实际含义 | 10.05²−101 = 0.0025，是方程残差 |

这是指标混用。可诚实地说该例格式更规范；原文显示的 pre 数字事实上更接近 sqrt(101)，post 则舍入为两位小数。无法用这个单例证明总体数值准确率提升，更不能把 +15 个百分点的 SFT 结果归到 GRPO。

§5 另一例的原生数学公式是 sqrt(x²+165) − sqrt(x²−52) = 7，样例答案 14。x=14 可满足，但该摘要没有额外限定正数时也允许 −14；它展示格式，不提供完备求解正确性验证。

### 语义路由器及论文定位

§3.2/4.2 说 Qwen3-Embedding-4B 4-bit 加载、LoRA r=16，针对 q_proj/v_proj；AFQMC/LCQMC 句对任务，用 CosineSimilarityLoss；PeftSentenceTransformer 处理 adapter-only 保存。论文没有提供路由分类准确率、分类标签定义、测试划分、和数学专家集成的日志或实际路由决策。它明确说两组件未集成，因此不应宣传“已完成 MoE 系统”。

DOCX 将 GRPO 展开为 Guided Reasoning and Problem Optimization，并称为 novel framework；PDF 另外把它写作 Good-Reject-Policy Optimization（p17/19）及 gradient-based policy optimization（p23）。这些都不是 TRL 所用算法名称。真实名称是 Group Relative Policy Optimization，来自 DeepSeekMath。新文档可描述“使用 GRPO 的格式与数值 reward 实验”，不宣称发明了 GRPO。此项已核对 [Hugging Face TRL 官方文档](https://huggingface.co/docs/trl/grpo_trainer) 和 [DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300)。

DOCX 参考文献大部分为 QA、BERT、SQuAD 等背景文献，缺少其关键 GRPO/LoRA/Unsloth/嵌入模型及数据集的对应引用；还留有 Appendix “Font, Format and Binding”等学校模板说明。新仓库应独立整理方法归属与引用，完整 DOCX 只作为历史材料保留。

## 公开仓库的作者与隐私处理

| 位置 | 内容 | 建议 |
|---|---|---|
| PDF p1/p5 | 两位作者的学号；作者姓名 | 保留作者署名，公开材料省略学号 |
| PDF p6 | 导师反馈、姓名和手写签名图片 | 原稿私人保留；公开项目一般不需要签名/评语页 |
| PDF p7 | 考官反馈、姓名和手写签名图片 | 同上 |
| DOCX 封面 p1 | 两位学号和导师姓名 | 公开摘要保留必要作者信息，省略学号 |
| DOCX docProps/core.xml | 编辑者个人邮箱、作者账户、编辑历史时间 | 不把原始 DOCX 元数据直接随公开草稿发布 |
| 全文提取和图像审计目录 | 包含上述原始信息与签名图 | 不放入 public-ready 包；由私人审计保留 |

不可把项目改为单人独立完成。两个封面与 PDF 致谢都明确显示 Yuzhen Guan 和 Weizhou Lu 合作；导师为 Thomas Canhao Xu。可在 README/论文引用中保留两位作者，个人贡献以可证明职责为准，不替作者编造分工。文件名 Group14 也与团队项目一致。

## 本次复现应建立的证据边界

1. 保存论文原表为“historical thesis-reported”，包括 n=20/20 和 +15 percentage points；保存 notebook 历史输出及本次运行结果为不同表，绝不强行调数字对齐。
2. 从实际代码恢复精确的数据入口、教师、完整 prompts、LoRA 参数和判分版本；对 PDF 与代码矛盾逐项标注。
3. 先恢复训练/推理可运行链路和 CPU 侧数据、解析、判分测试；只有真实模型推理或训练成功后，才声明对应 GPU 路径已验证。
4. 保留原始数据及来源记录；新清洗/去重/修正样本与原版分开，所有拆分固定随机种子并保留样本 ID。
5. 如果论文 40 题完整清单、原 checkpoint 或生成时 JSONL 不在提交材料中，应列为缺失证据。可建立新的可验证基准，但应明确不是复刻相同原始试验。
6. 以“数学题与竞赛式推理题的代码辅助求解”描述能力；未实现形式证明检查器时不宣传“证明正确率”。
7. 公开草稿保留双作者署名、代码/数据/上游 notebook 归属；论文原稿、含签名的图和提取全文不直接进入公开包。

## 审计工作文件

- pdf_full_text.txt 及 pdf_page_001.txt–pdf_page_042.txt：全 PDF 文本。
- docx_full_text.txt：按 466 个正文块编号的完整 DOCX 文本。
- docx_rendered_full_text.txt：本次 22 页渲染文本。
- extraction_manifest.json / image_manifest.json：结构和图像覆盖。
- docx_document.xml / docx_math_elements.xml：用于核查原生公式和文档结构。
- pdf_images、pdf_render、docx_render 及 contact images：视觉检查材料。
- docx_ancillary_text.txt / pdf_metadata.json：元数据与辅助部分检查，含私人信息，不公开。

