
# ========== 扩展23：大规模特征池搜索引擎 ==========
def patch_massive_search(period_range=None, n_trials=200):
    """激活大规模特征池搜索引擎。随机组合周期+附加模块，自动找到最优特征池。"""
    import numpy as np, random, time

    if period_range is None:
        period_range = (3, 25)

    def massive_search(self, X, y, periods=None, n=None):
        t = X[:, 0].astype(float)
        if periods is None: periods = list(range(*period_range))
        if n is None: n = n_trials

        extra_pool = ['poly2','poly3','lag_short','lag_long','int_exp','int_decay',
                      'tv_sin','tv_cos','cross_sin','cross_cos','sq_sin','sq_cos']
        best_r2, best_combo = -np.inf, None
        top_hits = []

        for i in range(n):
            T = random.choice(periods)
            features = [t, np.sin(2*np.pi*t/T), np.cos(2*np.pi*t/T)]
            n_extra = random.randint(0, min(5, len(extra_pool)))
            extras = random.sample(extra_pool, n_extra) if n_extra > 0 else []

            for c in extras:
                if c == 'poly2': features.append(t**2)
                elif c == 'poly3': features.append(t**3)
                elif c == 'lag_short': features.append(np.roll(t, random.choice([1,2,3])))
                elif c == 'lag_long': features.append(np.roll(t, random.choice([5,10,20])))
                elif c == 'int_exp': features.append(np.cumsum(t) * random.uniform(0.01, 0.3))
                elif c == 'int_decay': features.append(np.exp(-t/50))
                elif c == 'tv_sin': features.append(np.sin(2*np.pi*t/T) * t)
                elif c == 'tv_cos': features.append(np.cos(2*np.pi*t/T) * t)
                elif c == 'cross_sin': features.append(np.sin(2*np.pi*t/T) * np.sin(2*np.pi*t/11))
                elif c == 'cross_cos': features.append(np.cos(2*np.pi*t/T) * np.cos(2*np.pi*t/11))
                elif c == 'sq_sin': features.append(np.sin(2*np.pi*t/T)**2)
                elif c == 'sq_cos': features.append(np.cos(2*np.pi*t/T)**2)

            X_feat = np.column_stack(features)
            try:
                self.fit(X_feat, y, feature_names=[f'f{j}' for j in range(X_feat.shape[1])])
                r2 = self.r2_
                if r2 > best_r2: best_r2, best_combo = r2, (T, extras)
                if r2 > 0.3: top_hits.append((T, r2, extras[:4]))
            except: pass

        top_hits.sort(key=lambda x: -x[1])
        self.search_result_ = {'best_r2': best_r2, 'best_combo': best_combo, 'top_hits': top_hits[:10]}
        return self.search_result_

    CDE_V80.massive_search = massive_search
    print(f"[扩展23] 大规模特征池搜索引擎已激活")

def unpatch_massive_search():
    if hasattr(CDE_V80, 'massive_search'): delattr(CDE_V80, 'massive_search')
    print("[扩展23] 大规模搜索引擎已卸载")
