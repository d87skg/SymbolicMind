import json, io, numpy as np
from fastmcp import FastMCP

mcp = FastMCP("SymbolicMind", version="0.1.0")

@mcp.tool()
def discover_equation(data_csv: str, target_column: str, mode: str = "static") -> str:
    """从时间序列或静态数据中自动发现非线性动力学方程。"""
    import sys, os
    sys.path.insert(0, 'D:/CDE/SymbolicMind')
    from symbolimind.engine import CDE_V80
    df = np.genfromtxt(io.StringIO(data_csv), delimiter=',', names=True)
    col_names = list(df.dtype.names)
    if target_column not in col_names:
        return json.dumps({"error": f"'{target_column}' not found. Available: {col_names}"})
    feature_cols = [c for c in col_names if c != target_column]
    X = np.array([df[c] for c in feature_cols]).T
    y = df[target_column]
    cde = CDE_V80(time_series=(mode == "time_series"))
    cde.fit(X, y, feature_names=feature_cols)
    return json.dumps({
        "equation": cde.get_equation(),
        "r2": float(cde.r2_),
        "residual_autocorr": float(cde.residual_autocorr_),
        "p0_verdict": cde.get_p0_report()
    }, indent=2)

@mcp.tool()
def quickstart() -> str:
    """运行快速入门示例，发现简谐振动方程 d²x/dt² = -x"""
    t = np.linspace(0, 10, 200)
    x = np.cos(t)
    d2x = -np.cos(t)
    csv_data = "t,x,d2x_dt2\n"
    for i in range(len(t)):
        csv_data += f"{t[i]},{x[i]},{d2x[i]}\n"
    return discover_equation(csv_data, "d2x_dt2", "static")

if __name__ == "__main__":
    # 使用 SSE 传输模式（MCP 标准 HTTP 传输）
    mcp.run(transport="sse", port=9020)
