# 贡献指南

感谢您对 SymbolicMind 的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告 Bug
- 在 GitHub Issues 中提交，附上最小可复现示例。
- 请说明操作系统、Python 版本和依赖版本。

### 提出新功能
- 先开 Issue 讨论设计，再提交 PR。
- 新增功能需包含测试用例。

### 贡献算子库
- 在 `symbolimind/extensions.py` 或独立插件文件中实现新算子。
- 确保通过现有基准测试（Feynman、Strogatz 等）。
- 提交 PR 并描述算子的物理含义和适用场景。

### 添加基准
- 将数据集放入 `/data`，并在 `BENCHMARKS.md` 中更新成绩。
- 请确保数据集为 CSV 格式，包含列名。

### 代码风格
- 遵循 PEP 8，使用 `flake8` 或 `ruff` 检查。
- 变量命名清晰，关键函数需有 Docstring。

### 开发环境
```bash
pip install numpy scipy scikit-learn
pytest tests/

提交信息规范
格式：[类型] 描述

类型：feat, fix, docs, test, refactor 等

项目架构
SymbolicMind/
├── symbolimind/              # 核心引擎包（19项扩展）
│   ├── engine.py             # BIC 精拟合 + P0 探测器
│   ├── extensions.py         # 19 项固化扩展
│   ├── unified.py            # 统一接口
│   ├── skill.py              # AI 智能体工具接口
│   ├── brain.py              # 自然语言大脑
│   ├── memory.py             # 本地经验记忆库
│   └── grammar.py            # Physics Grammar Engine
├── tests/                    # 测试
├── colab/                    # Google Colab notebook
└── docs/                     # 文档

制度层
SymbolicMind 有三层不可修改的制度：

制度	核心原则
Scientific Constitution	可解释、可证伪、最小复杂度、NN不得裁决、残差优先视为未知规律
Physics Grammar Engine	操作符深度≤3、奇异性过滤、量纲一致性
Proposal Diversity Governance	30/30/30/10 多通道配额
任何贡献必须遵守以上制度。

行为准则
我们致力于建设一个开放、友好的社区，请保持尊重和包容。

许可证
所有贡献将在 MIT 许可下进行。


---

## ✅ 四份文档全部交付

| 文档 | 状态 |
|------|------|
| LIMITATIONS.md | ✅ 最终版已提供 |
| STRATEGIC_REPORT.md | ✅ 最终版已提供 |
| BENCHMARKS.md | ✅ 最终版已提供 |
| CONTRIBUTING.md | ✅ 最终版已提供 |

