import numpy as np

class PhysicsGrammarEngine:
    """
    Minimal Physics Grammar Engine.
    在候选表达式进入符号回归之前，执行物理合法性审查。
    """
    
    DIMENSION_MAP = {
        'θ': 'angle', 'θ₁': 'angle', 'θ₂': 'angle', 't1': 'angle', 't2': 'angle',
        'ω': 'angular_velocity', 'ω₁': 'angular_velocity', 'ω₂': 'angular_velocity',
        'o1': 'angular_velocity', 'o2': 'angular_velocity',
        'Δθ': 'angle', 'd': 'angle', 'sin_d': 'dimensionless',
        'cos_d': 'dimensionless', 'sin_t1': 'dimensionless', 'cos_t1': 'dimensionless',
        'sin_t2': 'dimensionless', 'cos_t2': 'dimensionless',
        'o1_sq': 'squared_velocity', 'o2_sq': 'squared_velocity',
    }
    
    def __init__(self, max_operator_depth=3, min_std=0.01):
        self.max_depth = max_operator_depth
        self.min_std = min_std
        self.validation_log = []
    
    def validate(self, expression_str, atom_values_dict, target_dimension='angular_velocity'):
        reasons = []
        
        # 1. 操作符深度检查
        depth = self._estimate_depth(expression_str)
        if depth > self.max_depth:
            reasons.append(f'OPERATOR_DEPTH: {depth} > {self.max_depth}')
        
        # 2. 奇异性检查
        if atom_values_dict is not None:
            try:
                values = atom_values_dict
                if isinstance(values, dict):
                    values = list(values.values())[0]
                if hasattr(values, '__len__'):
                    arr = np.asarray(values)
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        reasons.append('SINGULARITY: NaN or Inf in values')
                    elif np.std(arr) < self.min_std:
                        reasons.append(f'SINGULARITY: std={np.std(arr):.6f} < {self.min_std}')
            except:
                pass
        
        # 3. 量纲一致性检查
        dim_result = self._infer_dimension(expression_str)
        if dim_result and dim_result != target_dimension and dim_result != 'mixed' and dim_result != 'squared_velocity':
            reasons.append(f'DIMENSION: {dim_result} != {target_dimension}')
        
        is_valid = len(reasons) == 0
        self.validation_log.append({'expression': expression_str[:80], 'valid': is_valid, 'reasons': reasons})
        return is_valid, reasons
    
    def _estimate_depth(self, expr_str):
        depth = 0
        max_depth = 0
        for ch in expr_str:
            if ch == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ')':
                depth -= 1
        if 'sin' in expr_str or 'cos' in expr_str or 'sq' in expr_str:
            max_depth += 1
        return max_depth
    
    def _infer_dimension(self, expr_str):
        has_velocity = any(kw in expr_str for kw in ['o1', 'o2', 'ω₁', 'ω₂', 'o1_sq', 'o2_sq'])
        has_angle = any(kw in expr_str for kw in ['t1', 't2', 'θ₁', 'θ₂', 'd', 'sin_d', 'cos_d', 'sin_t1', 'cos_t1'])
        has_sq = any(kw in expr_str for kw in ['o1_sq', 'o2_sq'])
        
        if has_sq:
            return 'squared_velocity'
        elif has_velocity and has_angle:
            return 'mixed'
        elif has_velocity:
            return 'angular_velocity'
        elif has_angle:
            return 'angle'
        else:
            return 'dimensionless'
    
    def get_report(self):
        if not self.validation_log:
            return 'No validations performed.'
        total = len(self.validation_log)
        passed = sum(1 for v in self.validation_log if v['valid'])
        return f'Grammar Check: {passed}/{total} passed'


# ========== 测试 ==========
print('=== Minimal Physics Grammar Engine ===')
print()

grammar = PhysicsGrammarEngine(max_operator_depth=3, min_std=0.01)

test_cases = [
    ('sin_d * o1_sq', None, 'angular_velocity'),
    ('sin(cos(exp(t1)))', None, 'angular_velocity'),
    ('(d - d)', np.array([0.0, 0.0, 0.0]), 'angular_velocity'),
    ('t1 + o1', None, 'angular_velocity'),
    ('o1_sq * sin_d', None, 'squared_velocity'),
]

for expr, vals, target_dim in test_cases:
    is_valid, reasons = grammar.validate(expr, vals, target_dim)
    status = '✅' if is_valid else '❌'
    print(f'{status} {expr[:60]}')
    if not is_valid:
        for r in reasons:
            print(f'   → {r}')

print()
print(grammar.get_report())
