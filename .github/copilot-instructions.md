## 目的
为 AI 编程代理（例如 Copilot / 自动化助理）提供在此代码库中立刻可用的、可操作的上下文与运行线索，帮助快速定位关键资产、理解约定并安全修改代码或数据处理流程。

## 最重要的文件与位置（一眼能看懂项目）
- `rf.py`（仓库根）：主要训练/评估脚本（RandomForest），使用绝对路径读取外部 CSV，输出结果 CSV（注意路径硬编码）。
- `test/rf.py`：与根 `rf.py` 相似，但使用仓库内 `test/study_data_wna.csv`，并将结果写到 `test/result/`，是可执行的快速入口。
- `2025-02b/datapackage.json`：FoG 数据包的 schema 与资源列表（关键：数据模式、主键、外键、字段约束），是理解数据格式和验证的来源。
- `RGI2000-v7.0-G-02_western_canada_usa/` 和 `nsidc0770_02.rgi*/`：包含 RGI/GLIMS 矢量/CSV 数据与元数据，用于地理/冰川参考。

## 快速上手（针对 AI 代理）
1. 环境：本项目没有 `requirements.txt` 或 `pyproject.toml`。可推断的依赖来自导入：`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`。在 Windows PowerShell 下建议使用虚拟环境并安装这些包。
2. 运行示例：`test/rf.py` 是推荐的快速运行目标——它引用相对仓库路径（`H:/Code/SMB/test/study_data_wna.csv` 在当前工作区）。该脚本生成 `test/result/all_data_result_time_wna.csv` 和 `test/result/pred_result_time_wna.csv`。
3. 注意点：根 `rf.py` 使用了不同的绝对路径（例如 `H:\workspace4_ear5\...`），更改代码前优先在 `test/rf.py` 验证变更。

## 发现的工程约定与模式（可被代理直接利用）
- 数据优先：仓库包含大型数据包（`2025-02b`），数据结构以 `datapackage.json` 为准；任何数据修改应遵循该 schema（字段名、主键、外键）。
- 小脚本驱动：主要功能由自包含的 Python 脚本完成（无专门的包结构或 CLI）；agent 修改应尽量保持脚本独立、避免全局路径硬编码。
- 输出目录：脚本通常将结果写入 `test/result/`（用作 QA）；在修改或重构时保留或映射到该目录以便比对。

## 典型改动场景与建议实现策略
- 如果需要更可移植的运行：把硬编码的文件路径替换成命令行参数或相对路径；首选修改 `test/rf.py` 做为范例并运行验证。
- 导出/处理数据时，优先使用 `pandas.read_csv()` + `DataFrame.to_csv()`，保持与现有脚本一致的列名（参见 `features_columns` 列表）。
- 若需验证数据一致性，读取 `2025-02b/datapackage.json` 的 schema 字段用于自动校验（例如确保 `glacier.id` 是整数且存在）。

## 依赖与运行/调试流程（可直接给出给开发者的步骤）
- 在 Windows PowerShell 中创建虚拟环境并安装依赖（示例）:
  - 创建 venv：`python -m venv .venv`；激活：`.\.venv\Scripts\Activate.ps1`；安装：`pip install pandas numpy scikit-learn matplotlib scipy`。
- 运行快速验证：在激活的环境中运行 `python test/rf.py`，观察 `test/result/` 下 CSV 输出。

## 安全与数据注意事项
- 仓库包含大型/权威数据包（WGMS FoG、RGI）；不要在没有核对的情况下改写或覆盖 `2025-02b/data/*.csv`。
- 在自动补丁或 PR 中，agent 应避免把外部绝对路径写入到主分支（例如 `H:\workspace4_ear5\...`）；使用相对路径或配置文件。

## 可搜索的示例片段（供自动编辑时参考）
- 训练/评估循环：示例在 `test/rf.py` 中，从 `for test_year in range(1980, 2021):` 到 `model.predict()`。
- 输出路径示例：`results_df.to_csv('H:/Code/SMB/test/result/all_data_result_time_wna.csv', index=False)`。

## 变更合并策略（agent 操作须知）
- 若生成新的脚本或修改现有脚本：
  - 在 `test/` 下添加或验证改动（快速可复现）。
  - 包含一个短的 README 或注释，说明如何在本地运行（包括激活虚拟环境和依赖）。
  - 不要直接修改 `2025-02b` 数据文件；数据模型变更需要人工审查。

## 需要人工确认的地方（提醒 agent 报告）
- 是否将硬编码路径替换为配置或 CLI 参数（会影响所有脚本）。
- 是否需要添加 `requirements.txt` 或 CI（目前仓库无 CI 配置）。

---
如果你想要我把某一段（例如：把根 `rf.py` 的路径统一改为相对路径，或为项目生成 `requirements.txt`）自动实现并提交一个 PR，我可以继续执行。请告诉我你的优先项或任何遗漏的关键信息。
