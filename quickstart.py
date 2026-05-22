"""
SymbolicMind Quickstart
=======================
在 5 分钟内体验 AI 科学家从数据中发现物理定律。
"""

import numpy as np
from symbolimind.skill import cde_discover_equation
from symbolimind.brain import create_brain

print("=== SymbolicMind Quickstart ===")
print("正在生成简谐振动数据...")

# 1. 生成数据：d²x/dt² = -x
t = np.linspace(0, 10, 200)
x = np.cos(t)
dx_dt = -np.sin(t)
d2x_dt2 = -np.cos(t)

# 2. 构建 CSV 格式数据（模拟真实实验数据）
csv_data = "t,x,dx_dt,d2x_dt2\n"
for i in range(len(t)):
    csv_data += f"{t[i]},{x[i]},{dx_dt[i]},{d2x_dt2[i]}\n"

# 3. 调用引擎发现控制方程
print("正在发现 d²x/dt² 的控制方程...")
result = cde_discover_equation(csv_data, target_column="d2x_dt2", mode="static")

# 4. 输出原始结果
print("\n--- 引擎输出 ---")
print(f"方程: {result['equation']}")
print(f"R²: {result['r2']:.4f}")
print(f"残差自相关: {result['residual_autocorr']:.4f}")
print(f"P0 判决: {result['p0_verdict']}")

# 5. AI 科学家报告
brain = create_brain("local")
report = brain.generate_answer(result, target_column="d2x_dt2")
print("\n--- AI 科学家报告 ---")
print(report)

print("\n✅ Quickstart 完成。SymbolicMind 已就绪。")

