import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from symbolimind.skill import cde_discover_equation

# 创建测试数据
t = np.linspace(0, 10, 200)
x = np.cos(t)
dx_dt = -np.sin(t)
d2x_dt2 = -np.cos(t)

csv_data = 't,x,dx_dt,d2x_dt2\n'
for i in range(len(t)):
    csv_data += f'{t[i]},{x[i]},{dx_dt[i]},{d2x_dt2[i]}\n'

print('=== 第一次调用（写入记忆） ===')
result1 = cde_discover_equation(csv_data, 'd2x_dt2')
print(f'发现：{result1["equation"]}')
print()

print('=== 第二次调用（触发记忆） ===')
result2 = cde_discover_equation(csv_data, 'd2x_dt2')
print(f'发现：{result2["equation"]}')

