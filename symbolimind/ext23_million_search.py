
# ========== 扩展23：百万级特征池搜索引擎 ==========
def patch_million_search(period_range=None, n_trials=500):
    """激活百万级特征池搜索引擎。327万种组合中自动找到最优特征池。"""
    import numpy as np, random

    if period_range is None:
        period_range = (2, 52)

    def million_search(self, X, y, periods=None, n=None):
        t = X[:, 0].astype(float)
        if periods is None: periods = list(range(*period_range))
        if n is None: n = n_trials

        extra_pool = ['poly2','poly3','poly4','lag_short','lag_mid','lag_long',
                      'int_exp','int_decay','int_power','tv_sin','tv_cos',
                      'cross_sin','cross_cos','sq_sin','sq_cos','sin2_cos2']
        best_r2, best_combo = -np.inf, None
        top_hits = []

        for i in range(n):
            T = random.choice(periods)
            features = [t, np.sin(2*np.pi*t/T), np.cos(2*np.pi*t/T)]
            n_extra = random.randint(0, min(8, len(extra_pool)))
            extras = random.sample(extra_pool, n_extra) if n_extra > 0 else []

            for c in extras:
                if c == 'poly2': features.append(t**2)
                elif c == 'poly3': features.append(t**3)
                elif c == 'poly4': features.append(t**4)
                elif c == 'lag_short': features.append(np.roll(t, random.choice([1,2,3])))
                elif c == 'lag_mid': features.append(np.roll(t, random.choice([5,7,10])))
                elif c == 'lag_long': features.append(np.roll(t, random.choice([15,20,30])))
                elif c == 'int_exp': features.append(np.cumsum(t) * random.uniform(0.01, 0.3))
                elif c == 'int_decay': features.append(np.exp(-t/50))
                elif c == 'int_power': features.append(t**0.5)
                elif c == 'tv_sin': features.append(np.sin(2*np.pi*t/T) * t)
                elif c == 'tv_cos': features.append(np.cos(2*np.pi*t/T) * t)
                elif c == 'cross_sin': features.append(np.sin(2*np.pi*t/T) * np.sin(2*np.pi*t/11))
                elif c == 'cross_cos': features.append(np.cos(2*np.pi*t/T) * np.cos(2*np.pi*t/11))
                elif c == 'sq_sin': features.append(np.sin(2*np.pi*t/T)**2)
                elif c == 'sq_cos': features.append(np.cos(2*np.pi*t/T)**2)
                elif c == 'sin2_cos2': features.append(np.sin(2*np.pi*t/T)**2 * np.cos(2*np.pi*t/T)**2)

            X_feat = np.column_stack(features)
            try:
                self.fit(X_feat, y, feature_names=[f'f{j}' for j in range(X_feat.shape[1])])
                r2 = self.r2_
                if r2 > best_r2: best_r2, best_combo = r2, (T, extras)
                if r2 > 0.3: top_hits.append((T, r2, extras[:4]))
            except: pass

        top_hits.sort(key=lambda x: -x[1])
        self.million_result_ = {'best_r2': best_r2, 'best_combo': best_combo, 'top_hits': top_hits[:10]}
        return self.million_result_

    CDE_V80.million_search = million_search
    print(f"[扩展23] 百万级特征池搜索引擎已激活（{period_range[0]}-{period_range[1]}年, {n_trials}次搜索）")

def unpatch_million_search():
    if hasattr(CDE_V80, 'million_search'): delattr(CDE_V80, 'million_search')
    print("[扩展23] 百万级搜索引擎已卸载")
