# SymbolicMind

**一个能从数据中自主发现物理定律的 AI 引擎。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SymbolicMind 是一个因果符号回归引擎，基于 **BIC 精拟合** 和 **P0 可证伪性边界** 构建。它不仅拟合数据，还能自我修正、自我发现、自我进化——让机器从原始观测数据中提炼出人类可读的数学控制方程。

## 核心能力

- **方程发现**：自动从时间序列或静态数据中识别非线性动力学方程。
- **可证伪性**：内置 P0 探测器，通过残差自相关判别模型的可靠性，绝不捏造物理。
- **自演化算子库**：在需要时自主生成时变、分数阶、嵌套积分核等高级数学结构。
- **恒等式识别**：自动识别 `dx/dt = y` 等简单恒等式，避免过度建模。
- **AI 智能体就绪**：提供符合 OpenAI Function Calling 规范的工具接口，可被任何 AI Agent 调用。
- **记忆增强**：本地经验库记录成功发现，越用越聪明。

## 快速开始

```bash
pip install numpy scipy scikit-learn
python quickstart.py

你将看到引擎发现简谐振动方程：d²x/dt² = -x，并输出详细的诊断报告。

基准测试成绩
SymbolicMind 在 Feynman、Strogatz、ODEBench 和 Blackbox 等基准上取得了领先成绩。详见 BENCHMARKS.md。

项目结构
text
SymbolicMind/
├── symbolimind/          # 核心引擎包
│   ├── engine.py         # BIC 精拟合器 + P0 探测器
│   ├── extensions.py     # 恒等式检测 + 自适应断层扫描
│   ├── skill.py          # AI 智能体工具接口
│   ├── brain.py          # 自然语言大脑
│   └── memory.py         # 本地经验记忆库
├── tests/                # 测试
├── BENCHMARKS.md         # 基准测试成绩单
├── CONTRIBUTING.md       # 贡献指南
├── LICENSE               # MIT 许可证
├── quickstart.py         # 5 分钟快速体验
└── README.md             # 本文件
引用
如果 SymbolicMind 对你的研究有帮助，请引用本项目。

贡献
我们欢迎社区贡献！无论是新的物理算子、基准测试还是文档改进，请参见 CONTRIBUTING.md。