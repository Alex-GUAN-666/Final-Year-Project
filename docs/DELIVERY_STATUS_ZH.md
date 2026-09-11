# 交付状态与下一步

更新时间：2026-09-11。

## 已完成

- 完整恢复代码、数据、原始研究证据与运行文档。
- `docs/METHODOLOGY.md` 解释教师分工、数据筛选、模板、损失、LoRA 参数、评分与历史/新实验差异。
- 一键核验检查 18 个源文件哈希、历史 12/20 和 15/20、5/20 重合题、436 个候选与 187 道评测题的逐字节重建；27 项测试通过。
- 小型随机 Qwen3 已实际完成两步 CPU 训练、LoRA 合并与推理；它不是 4B 模型成绩。
- 新增历史程序重放入口：核对 40 条历史回复，其中 36 条包含可提取程序。仅输入溯源核验已执行，Docker 重放尚未执行。

## 仍未完成

1. GitHub 完整上传：用户已明确授权公开上传全部文件到 `Alex-GUAN-666/Final-Year-Project` 的 `main`。本地 git push 没有登录凭据；连接器创建完整文件树时，自动审核报“Codex ran out of room in the model context window”，并要求在新对话中继续。没有绕过审核。远端仍是初始提交。
2. GitHub Actions / 实际 Docker 测试：工作流已写好，但因上传被拦截没有启动，不可宣称通过。
3. 完整 Qwen3-4B GPU 训练与推理：当前环境只有 CPU。原微调权重未恢复，因此不能保证重新训练逐题/总分精确等于历史成绩。
4. 新的无规范化题目重合划分尚无 4B 评测成绩。历史推理分数有 5/20 训练重合；算术 70%→85% 只有论文报告，未恢复配对预测。

## 下载后先核验

解压并在项目根目录运行：

```bash
python scripts/verify_reconstruction.py --report outputs/reviewer_check.json
python scripts/replay_historical_programs.py --check-inputs
```

Python 3.12 已验证。详细安装与 GPU/Docker 命令见 `docs/RUNNING.md` 和 README。报告路径必须是新路径。

## 新对话继续上传时的交接

使用这个交付包，不要退回较早的 24 测试版本。保留当前 27 测试版本中的评分证据复核补丁、方法论和两个核验入口。明确目标仓库和完整公开上传授权后，通过正常审批上传；成功后从远端重新克隆验证并运行云端三项任务。历史程序重放如产生差异，保留差异并解释，不修改历史成绩来匹配。仅在观察到实际成功后更新 GitHub/CI/Docker 状态。
