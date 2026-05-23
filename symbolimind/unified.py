import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'D:/CDE/SymbolicMind')

import numpy as np

class SymbolicMind:
    def __init__(self, load_extensions=False):
        self.cde = None
        if load_extensions:
            from symbolimind.extensions import apply_all_patches
            apply_all_patches()
        from symbolimind.engine import CDE_V80
        self.cde = CDE_V80(time_series=False)
    
    def discover(self, data_csv=None, X=None, y=None, target_column=None, mode='static', 
                 use_memory=True, verbose=True):
        if data_csv is not None:
            df = np.genfromtxt(io.StringIO(data_csv), delimiter=',', names=True)
            col_names = df.dtype.names
            if target_column is None:
                target_column = col_names[-1]
            if target_column not in col_names:
                return {'error': f'Target column "{target_column}" not found. Available: {list(col_names)}'}
            feature_cols = [c for c in col_names if c != target_column]
            X = np.array([df[c] for c in feature_cols]).T
            y = df[target_column]
            feature_names = feature_cols
        elif X is not None and y is not None:
            feature_names = [f'x{i}' for i in range(X.shape[1])]
        else:
            return {'error': '请提供 data_csv 或 X+y'}
        
        if use_memory and data_csv:
            try:
                from symbolimind.memory import CDEMemory
                mem = CDEMemory()
                remembered = mem.recall(data_csv)
                if remembered:
                    if verbose: print('记忆触发：复用上次成功经验')
                    return remembered
            except: pass
        
        cde = self.cde
        cde.time_series = (mode == 'time_series')
        cde.fit(X, y, feature_names=feature_names)
        
        result = {
            'equation': cde.get_equation(),
            'r2': cde.r2_,
            'residual_autocorr': cde.residual_autocorr_,
            'p0_verdict': cde.get_p0_report(),
        }
        
        if use_memory and data_csv and result['r2'] > 0.95:
            try:
                from symbolimind.memory import CDEMemory
                mem = CDEMemory()
                mem.memorize(data_csv, target_column, result)
                if verbose: print('记忆更新：成功经验已存储')
            except: pass
        
        if verbose:
            print(f'发现方程: {result["equation"]}')
            print(f'R2: {result["r2"]:.4f}')
            print(f'P0: {result["p0_verdict"]}')
        
        return result
    
    def quickstart(self):
        t = np.linspace(0, 10, 200)
        x = np.cos(t)
        dx_dt = -np.sin(t)
        d2x_dt2 = -np.cos(t)
        csv_data = "t,x,dx_dt,d2x_dt2\n"
        for i in range(len(t)):
            csv_data += f"{t[i]},{x[i]},{dx_dt[i]},{d2x_dt2[i]}\n"
        return self.discover(data_csv=csv_data, target_column='d2x_dt2', mode='static')


if __name__ == '__main__':
    print('=== SymbolicMind 统一接口测试 ===')
    sm = SymbolicMind(load_extensions=False)
    result = sm.quickstart()
    print(f'\n统一接口工作正常')
