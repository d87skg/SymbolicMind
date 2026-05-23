import json, sys, io, os, numpy as np

# ========== 强制路径初始化（解决 Claude Desktop 环境变量问题） ==========
SYMBOLICMIND_ROOT = os.path.dirname(os.path.abspath(__file__))
if SYMBOLICMIND_ROOT not in sys.path:
    sys.path.insert(0, SYMBOLICMIND_ROOT)
if os.path.dirname(SYMBOLICMIND_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(SYMBOLICMIND_ROOT))

# ========== 工具执行函数 ==========
def execute_tool(tool_name, arguments):
    try:
        if tool_name == "discover_equation":
            return run_discovery(arguments)
        elif tool_name == "quickstart":
            return run_quickstart()
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def run_discovery(args):
    data_csv = args.get("data_csv", "")
    target = args.get("target_column", "")
    mode = args.get("mode", "static")
    
    df = np.genfromtxt(io.StringIO(data_csv), delimiter=',', names=True)
    col_names = list(df.dtype.names)
    
    if target not in col_names:
        return json.dumps({"error": f"'{target}' not found. Available: {col_names}"})
    
    feature_cols = [c for c in col_names if c != target]
    X = np.array([df[c] for c in feature_cols]).T
    y = df[target]
    
    from symbolimind.engine import CDE_V80
    cde = CDE_V80(time_series=(mode == "time_series"))
    cde.fit(X, y, feature_names=feature_cols)
    
    return json.dumps({
        "equation": cde.get_equation(),
        "r2": float(cde.r2_),
        "residual_autocorr": float(cde.residual_autocorr_),
        "p0_verdict": cde.get_p0_report()
    }, indent=2)

def run_quickstart():
    t = np.linspace(0, 10, 200)
    x = np.cos(t)
    d2x = -np.cos(t)
    csv_data = "t,x,d2x_dt2\n"
    for i in range(len(t)):
        csv_data += f"{t[i]},{x[i]},{d2x[i]}\n"
    return run_discovery({"data_csv": csv_data, "target_column": "d2x_dt2", "mode": "static"})

# ========== MCP 主循环 ==========
if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # 向 stderr 写入启动确认（用于调试 Claude Desktop 日志）
    sys.stderr.write(f"MCP Server starting from {os.getcwd()}\n")
    sys.stderr.flush()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line or line.strip() == "":
                continue
            
            request = json.loads(line.strip())
            method = request.get("method", "")
            req_id = request.get("id", 0)
            
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "symbolimind", "version": "0.1.0"},
                        "capabilities": {"tools": {}}
                    }
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "discover_equation",
                                "description": "从时间序列或静态数据中自动发现非线性动力学方程。",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "data_csv": {"type": "string"},
                                        "target_column": {"type": "string"},
                                        "mode": {"type": "string", "enum": ["static", "time_series"], "default": "static"}
                                    },
                                    "required": ["data_csv", "target_column"]
                                }
                            },
                            {
                                "name": "quickstart",
                                "description": "运行快速入门示例，发现简谐振动方程",
                                "inputSchema": {"type": "object", "properties": {}}
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                tool_name = request.get("params", {}).get("name", "")
                arguments = request.get("params", {}).get("arguments", {})
                result_text = execute_tool(tool_name, arguments)
                response = {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": result_text}]}
                }
            else:
                response = {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
            
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            
        except json.JSONDecodeError:
            continue
        except EOFError:
            break
        except Exception as e:
            sys.stderr.write(f"MCP Error: {str(e)}\n")
            sys.stderr.flush()
