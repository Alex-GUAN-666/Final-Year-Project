# 交付与云端验证状态

更新时间：2026-09-11。项目已上传至[公开 GitHub 仓库](https://github.com/Alex-GUAN-666/Final-Year-Project)。

**已有云端验证快照全部通过：**[run 34580694788](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/runs/34580694788)，测试[代码版本 `cfac02190ff74c00c8facdb81b863b36c5f1981c`](https://github.com/Alex-GUAN-666/Final-Year-Project/commit/cfac02190ff74c00c8facdb81b863b36c5f1981c)。[完整验证说明](CLOUD_VALIDATION.md)与[验证记录](../reports/cloud/2026-09-11/verification.json)保留检查范围和证据。下表描述该次运行，不覆盖后来新增 notebook 的核查。最新工作流状态以仓库 Actions 为准。

## 面试官可以独立核验什么

| 检查 | 已观察到的结果 |
|---|---|
| 文件与数据 | 18 个源文件/公开副本哈希通过；436 个训练候选、187 道评估题逐字节重建一致；规范化题目交集为 0 |
| 单元测试 | 32 项通过，包括数据、评分、比较与 Docker 清理回归检查 |
| 真实训练链路 | 随机小型 Qwen3 在 CPU 上完成 2 步训练、adapter 保存、合并、重新加载与推理 |
| 真实 Docker 执行 | 8 项集成检查通过，覆盖隔离执行、失败处理、延后评分与成对比较 |
| 历史程序重放 | 40 条回复全部评分，包含 36 段程序和 4 条无代码回复；base 12/20、merged 15/20，逐题判分与所有可比较 stdout 均匹配，退出码 0 |

前两次云端运行发现 Docker 清理竞态和集成测试的 `Path` 参数问题。修复后补充 5 项清理回归测试，最终三项云端任务全部成功。历史数据、模型回复和历史分数没有为通过检查而改写。

方法论、教师分工、数据筛选、训练损失、LoRA 参数和评分规则见 [METHODOLOGY.md](METHODOLOGY.md)。本地 tokenizer 检查重建了历史 452 条与新实验 372 条训练样本；云端工作流没有重新下载 tokenizer，这两类证据分别记录。

## 新增历史材料

本次核查了 16 份新增 FYP notebook 版本，选取 6 份关键记录纳入仓库；其余版本保留在原上传压缩包中。重复代码、不同运行和提示词设置见 [RECOVERED_NOTEBOOKS.md](RECOVERED_NOTEBOOKS.md)。新增 CoT 顺序执行记录支持 base 12/20、merged 15/20；旧 merged notebook 继续保留完整回答。四份算术记录均为同一组 20 题的 17/20，但实际加载 `qwen-3/transformers/4b/1`，不是项目 merged 或训练基础模型 `4b-base/1`。这些是静态核查的历史记录，不是新完成的 4B 实验。

## 尚未完成的模型实验

完整 Qwen3-4B GPU 训练和推理尚未重新运行，原微调权重没有恢复。云端小模型训练与历史程序重放不能证明重新训练一定得到原分数，也没有产生新的 4B 准确率。

历史推理 60% → 75% 仍有 5/20 重建训练重合。算术 70% → 85% 仍是未经配对记录确认的论文报告；新找回的另一模型 17/20 记录不能替代这项证据。模型资源名 `qwen3-4b-sfft-merged` 和 `modelId=508779` 仅提供查找线索，原权重仍未恢复。新的 187 题划分尚无完成的 4B 基准结果。

## 下载后核验

在项目根目录、Python 3.12 环境运行：

```bash
python scripts/verify_reconstruction.py --report outputs/reviewer_check.json
python scripts/replay_historical_programs.py --check-inputs
```

这两个命令不需要 GPU 或模型下载，也不在宿主机执行生成程序。实际 CPU 训练、Docker 重放和固定版本检出命令见 [CLOUD_VALIDATION.md](CLOUD_VALIDATION.md)；完整 GPU 环境与运行命令见 [RUNNING.md](RUNNING.md)。报告路径必须是新路径。
