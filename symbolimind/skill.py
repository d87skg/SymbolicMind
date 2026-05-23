"""
SymbolicMind Skill - AI Agent Tool

Wraps the CDE engine into a callable function for AI agents.
Supports Function Calling schema and memory-enhanced discovery.

Author: [Your Name]
License: MIT
"""
import sys, io, json
import numpy as np

# ---- 自动加载 V8.1 扩展 ----

from symbolimind.engine import CDE_V80
from symbolimind.memory import CDEMemory

# ---- 初始化记忆 ----
memory = CDEMemory(enabled=True)

def cde_discover_equation(data_csv, target_column, mode="static", use_memory=True):
    """CDE 方程发现技能（带记忆增强）。"""
    # 1. 回忆
    if use_memory:
        remembered = memory.recall(data_csv)
        if remembered:
            print("🧠 记忆触发：复用上次成功经验")
            return remembered
    
    # 2. 常规发现
    df = np.genfromtxt(io.StringIO(data_csv), delimiter=',', names=True)
    col_names = df.dtype.names
    if target_column not in col_names:
        return {"error": f"Target column '{target_column}' not found. Available: {list(col_names)}"}
    
    feature_cols = [c for c in col_names if c != target_column]
    X = np.array([df[c] for c in feature_cols]).T
    y = df[target_column]
    
    cde = CDE_V80(time_series=(mode == "time_series"))
    cde.fit(X, y, feature_names=feature_cols)
    
    result = {
        "equation": cde.get_equation(),
        "r2": cde.r2_,
        "residual_autocorr": cde.residual_autocorr_,
        "p0_verdict": cde.get_p0_report()
    }
    
    # 3. 记忆：只要R²>0.95就存储（放宽条件）
    if use_memory and result["r2"] > 0.95:
        memory.memorize(data_csv, target_column, result)
        print("🧠 记忆更新：成功经验已存储。")
    
    return result
