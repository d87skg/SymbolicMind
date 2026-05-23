
# ========== 扩展22：周期自动发现引擎 ==========
def patch_period_discovery(period_candidates=None):
    """激活周期自动发现引擎。"""
    import numpy as np

    if period_candidates is None:
        period_candidates = [5, 7, 9, 10, 11, 12, 13, 15, 18, 22]

    def period_scan(self, X, y, periods=None, feature_names=None):
        t = X[:, 0].astype(float)
        best_r2, best_period, best_equation = -np.inf, None, None

        if periods is None:
            periods = period_candidates

        for T in periods:
            X_features = np.column_stack([
                t, np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T),
            ])
            self.fit(X_features, y, feature_names=['t', f'sin(2πt/{T})', f'cos(2πt/{T})'])
            if self.r2_ > best_r2:
                best_r2, best_period, best_equation = self.r2_, T, self.get_equation()

        self.period_result_ = {'best_period': best_period, 'best_r2': best_r2, 'best_equation': best_equation}
        return best_period, best_r2, best_equation

    CDE_V80.period_scan = period_scan
    print(f"[扩展] 周期自动发现引擎已激活")

def unpatch_period_discovery():
    if hasattr(CDE_V80, 'period_scan'): delattr(CDE_V80, 'period_scan')
    print("[扩展] 周期自动发现引擎已卸载")
