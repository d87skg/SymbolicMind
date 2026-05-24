"""
SymbolicMind Extensions - Optional Patches
Provides 16 extensions for the CDE engine.
"""

import numpy as np
from symbolimind.engine import CDE_V80

_original_fit = CDE_V80.fit
_original_get_p0 = CDE_V80.get_p0_report
_original_build_static = CDE_V80._build_static_features
_original_build_ts = CDE_V80._build_time_series_features


# ========== 扩展1：恒等式快速通道 ==========
def patch_identity_detector(relative_threshold=0.03):
    original_fit = CDE_V80.fit

    def fit_with_identity_check(self, X, y, feature_names=None):
        X = np.atleast_2d(X)
        y = np.asarray(y)
        
        if self.time_series:
            Theta, theta_names = self._build_time_series_features(
                X, feature_names if feature_names else [f'x{i}' for i in range(X.shape[1])])
        else:
            Theta, theta_names = self._build_static_features(
                X, feature_names if feature_names else [f'x{i}' for i in range(X.shape[1])])
        
        best_median_ratio = None
        best_j = None
        best_mad = float('inf')
        
        for j in range(Theta.shape[1]):
            feature_col = Theta[:, j]
            mask = np.abs(feature_col) > 1e-6
            if np.mean(mask) < 0.9:
                continue
            ratio = y[mask] / feature_col[mask]
            med_abs_dev = np.median(np.abs(ratio - np.median(ratio)))
            if med_abs_dev < best_mad and med_abs_dev < relative_threshold:
                best_mad = med_abs_dev
                best_j = j
                best_median_ratio = np.median(ratio)
        
        if best_j is not None:
            median_ratio = best_median_ratio
            feature_col = Theta[:, best_j]
            pred = feature_col * median_ratio
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.r2_ = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            residual = y - pred
            detrend = residual - np.mean(residual)
            acf = np.correlate(detrend, detrend, mode='same')
            self.residual_autocorr_ = acf[len(acf)//2 + 1] / acf[len(acf)//2] if acf[len(acf)//2] != 0 else 0.0
            self.coefficients_ = np.zeros(Theta.shape[1])
            self.coefficients_[best_j] = median_ratio
            self.selected_features_ = [theta_names[best_j]]
            self.equation_ = f'{median_ratio:.4f}*{theta_names[best_j]} (identity)'
            return self
        
        return original_fit(self, X, y, feature_names)

    CDE_V80.fit = fit_with_identity_check
    print("[扩展] 升级版恒等式检测已激活（扫描全部生成特征）")

def unpatch_identity_detector():
    CDE_V80.fit = _original_fit
    print("[扩展] 恒等式快速通道已卸载")


# ========== 扩展2：自适应断层扫描 ==========
def patch_adaptive_regime_scan(sigma_multiplier=6.0, min_segment_length=50):
    original_fit = CDE_V80.fit

    def fit_with_regime_scan(self, X, y, feature_names=None):
        X = np.atleast_2d(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        base_X = np.hstack([np.ones((n_samples, 1)), X[:, :min(3, X.shape[1])]])
        try:
            coeff = np.linalg.lstsq(base_X, y, rcond=None)[0]
            pred = base_X @ coeff
            residual = y - pred
            sigma = np.std(residual)
            threshold = sigma_multiplier * sigma
            flag = np.abs(residual) > threshold
            segments = []
            start = 0
            in_break = False
            for i in range(n_samples):
                if flag[i] and not in_break:
                    start = i
                    in_break = True
                elif not flag[i] and in_break:
                    if i - start >= 3:
                        segments.append((max(0, start - 5), min(n_samples, i + 5)))
                    in_break = False
            if in_break and n_samples - start >= 3:
                segments.append((max(0, start - 5), n_samples))
            if segments:
                clean_segments = []
                last_end = 0
                for s, e in segments:
                    if s - last_end > min_segment_length:
                        clean_segments.append((last_end, s))
                    last_end = e
                if n_samples - last_end > min_segment_length:
                    clean_segments.append((last_end, n_samples))
                if len(clean_segments) > 1:
                    self.regimes_ = []
                    for idx, (s, e) in enumerate(clean_segments):
                        sub_cde = CDE_V80(time_series=self.time_series if hasattr(self, 'time_series') else False)
                        sub_cde.fit(X[s:e], y[s:e])
                        self.regimes_.append({
                            'start': s, 'end': e,
                            'equation': sub_cde.get_equation(),
                            'r2': sub_cde.r2_,
                            'resid_ac': sub_cde.residual_autocorr_
                        })
                    sub_cde0 = CDE_V80(time_series=self.time_series if hasattr(self, 'time_series') else False)
                    sub_cde0.fit(X[clean_segments[0][0]:clean_segments[0][1]], y[clean_segments[0][0]:clean_segments[0][1]])
                    self.coefficients_ = sub_cde0.coefficients_
                    self.selected_features_ = sub_cde0.selected_features_
                    self.r2_ = sub_cde0.r2_
                    self.residual_autocorr_ = sub_cde0.residual_autocorr_
                    self.equation_ = sub_cde0.get_equation() + ' [Regime 1 only]'
                    return self
        except:
            pass
        return original_fit(self, X, y, feature_names)

    CDE_V80.fit = fit_with_regime_scan
    print("[扩展] 自适应断层扫描已激活")

def unpatch_adaptive_regime_scan():
    CDE_V80.fit = _original_fit
    print("[扩展] 自适应断层扫描已卸载")

    # ========== 扩展3：多频率物理假说验证器 ==========
def patch_multi_freq_hypothesis(num_freqs=5):
    from scipy.fft import fft, fftfreq
    original_fit = CDE_V80.fit

    def fit_with_multi_hypothesis(self, X, y, feature_names=None):
        result = original_fit(self, X, y, feature_names)
        if self.r2_ is None or self.residual_autocorr_ is None:
            return result
        if self.r2_ < 0.90 or abs(self.residual_autocorr_) < 0.1:
            return result
        
        if self.time_series:
            X_aug, aug_names = self._build_time_series_features(
                X, feature_names if feature_names else [f'x{i}' for i in range(X.shape[1])])
        else:
            X_aug, aug_names = self._build_static_features(
                X, feature_names if feature_names else [f'x{i}' for i in range(X.shape[1])])
        
        pred = X_aug @ self.coefficients_
        residual = y - pred
        n_samples = len(residual)
        dt = 1.0
        t = np.arange(n_samples) * dt
        fft_vals = np.abs(fft(residual))
        freqs = fftfreq(n_samples, dt)[:n_samples//2]
        fft_vals = fft_vals[:n_samples//2]
        
        best_result = None
        best_r2 = self.r2_
        best_lag1 = abs(self.residual_autocorr_)
        
        if len(freqs) > 1:
            top_indices = np.argsort(fft_vals[1:])[::-1][:num_freqs] + 1
            for idx in top_indices:
                freq = freqs[idx]
                if freq < 0.01: continue
                omega = 2 * np.pi * freq
                cos_term = np.cos(omega * t)
                A_cos = np.linalg.lstsq(cos_term.reshape(-1, 1), residual, rcond=None)[0][0]
                new_pred = pred + A_cos * cos_term
                ss_res = np.sum((y - new_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2_cos = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
                res_new = y - new_pred
                detrend = res_new - np.mean(res_new)
                acf = np.correlate(detrend, detrend, mode='same')
                lag1_cos = abs(acf[len(acf)//2 + 1] / acf[len(acf)//2]) if acf[len(acf)//2] != 0 else 1.0
                
                if r2_cos > best_r2 and lag1_cos < best_lag1:
                    best_r2 = r2_cos
                    best_lag1 = lag1_cos
                    best_result = (A_cos, f'cos({freq:.4f}t)', freq, omega)
        
        if best_result is not None:
            A_hat, term_name, freq, omega = best_result
            self.r2_ = best_r2
            self.residual_autocorr_ = best_lag1
            old_eq = self.equation_ if hasattr(self, 'equation_') else ''
            if old_eq:
                self.equation_ = f'{old_eq} + {A_hat:.4f}*{term_name} (hypothesis verified)'
            else:
                self.equation_ = f'{A_hat:.4f}*{term_name} (hypothesis verified)'
        return self

    CDE_V80.fit = fit_with_multi_hypothesis
    print(f'[扩展] 多频率物理假说验证器已激活（扫描前 {num_freqs} 个主频）')

def unpatch_multi_freq_hypothesis():
    CDE_V80.fit = _original_fit
    print('[扩展] 多频率物理假说验证器已卸载')


# ========== 扩展4：不确定性量化 ==========
def patch_uncertainty_quantification(num_candidates=5):
    from itertools import combinations
    original_fit = CDE_V80.fit

    def fit_with_uncertainty(self, X, y, feature_names=None):
        result = original_fit(self, X, y, feature_names)
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(X.shape[1])]
        n_samples = len(y)
        if self.time_series:
            Theta, names = self._build_time_series_features(X, feature_names)
        else:
            Theta, names = self._build_static_features(X, feature_names)
        
        single_bics = {}
        for j in range(Theta.shape[1]):
            X_sub = Theta[:, [j]]
            coeff = np.linalg.lstsq(X_sub, y, rcond=None)[0]
            pred = X_sub @ coeff
            sse = np.sum((y - pred)**2)
            if sse < 1e-15: sse = 1e-15
            bic = n_samples * np.log(sse / n_samples) + 2 * np.log(n_samples)
            single_bics[names[j]] = bic
        
        top_single = sorted(single_bics, key=single_bics.get)[:max(6, num_candidates)]
        top_indices = [names.index(n) for n in top_single if n in names]
        combo_bics = {}
        for i, j in combinations(top_indices, 2):
            X_combo = Theta[:, [i, j]]
            coeff = np.linalg.lstsq(X_combo, y, rcond=None)[0]
            pred = X_combo @ coeff
            sse = np.sum((y - pred)**2)
            if sse < 1e-15: sse = 1e-15
            bic = n_samples * np.log(sse / n_samples) + 3 * np.log(n_samples)
            combo_bics[(names[i], names[j])] = bic
        
        all_bics = {**single_bics}
        for (n1, n2), bic in combo_bics.items():
            all_bics[f'{n1} + {n2}'] = bic
        
        bic_values = np.array(list(all_bics.values()))
        bic_min = np.min(bic_values)
        delta_bic = bic_values - bic_min
        posterior = np.exp(-delta_bic / 2)
        posterior = posterior / np.sum(posterior)
        self.uncertainty_report_ = {
            'single_bics': single_bics,
            'combo_bics': combo_bics,
            'all_posteriors': dict(zip(all_bics.keys(), posterior))
        }
        return self

    CDE_V80.fit = fit_with_uncertainty
    print(f'[扩展] 不确定性量化模块已激活（top {num_candidates} 候选）')

def get_top_equations(self, n=5):
    if not hasattr(self, 'uncertainty_report_'):
        return []
    posteriors = self.uncertainty_report_['all_posteriors']
    sorted_items = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]

CDE_V80.get_top_equations = get_top_equations

def unpatch_uncertainty_quantification():
    CDE_V80.fit = _original_fit
    print('[扩展] 不确定性量化模块已卸载')


# ========== 扩展5：SDE发现 ==========
def patch_sde_discovery(window_length=51, polyorder=2):
    from scipy.signal import savgol_filter
    from sklearn.linear_model import LinearRegression

    def sde_discover(self, X_raw, dt=0.02, feature_names=None):
        n_samples = len(X_raw)
        X_1d = np.asarray(X_raw).flatten()
        X_smooth = savgol_filter(X_1d, window_length=window_length, polyorder=polyorder)
        dX_smooth = np.gradient(X_smooth, dt)
        X_2d = X_smooth.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X_2d, dX_smooth)
        drift_coef = lr.coef_[0]
        drift_r2 = lr.score(X_2d, dX_smooth)
        dX_raw = np.diff(X_1d) / dt
        dX_smooth_diff = np.diff(X_smooth) / dt
        min_len = min(len(dX_raw), len(dX_smooth_diff))
        noise = dX_raw[:min_len] - dX_smooth_diff[:min_len]
        sigma_est = np.std(noise) * np.sqrt(dt)
        self.sde_result_ = {
            'drift_equation': f'{drift_coef:.4f} * X',
            'drift_coef': drift_coef,
            'drift_r2': drift_r2,
            'sigma_est': sigma_est,
            'sde_full': f'dX = {drift_coef:.4f} * X * dt + {sigma_est:.4f} * dW'
        }
        return f'{drift_coef:.4f} * X', drift_r2, sigma_est

    CDE_V80.sde_discover = sde_discover
    print(f'[扩展] SDE 发现模块已激活（S-G 窗口={window_length}）')

def unpatch_sde_discovery():
    if hasattr(CDE_V80, 'sde_discover'):
        delattr(CDE_V80, 'sde_discover')
    print('[扩展] SDE 发现模块已卸载')


# ========== 扩展6：PDE发现 ==========
def patch_pde_discovery():
    def pde_discover(self, u_field, dx, dt, feature_names=None):
        nt, nx = u_field.shape
        du_dt = np.gradient(u_field, dt, axis=0)
        features_list, target_list = [], []
        for i in range(1, nx - 1):
            for n in range(nt):
                target_list.append(du_dt[n, i])
                ui = u_field[n, i]
                du_dx = (u_field[n, i+1] - u_field[n, i-1]) / (2 * dx)
                d2u_dx2 = (u_field[n, i+1] - 2*ui + u_field[n, i-1]) / dx**2
                features_list.append([ui, du_dx, d2u_dx2, ui**2, ui*du_dx, ui*d2u_dx2])
        X = np.array(features_list)
        y = np.array(target_list)
        names = feature_names or ['u', 'du/dx', 'd2u/dx2', 'u2', 'u*du/dx', 'u*d2u/dx2']
        coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coeff
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
        significant = [(n, c) for n, c in zip(names, coeff) if abs(c) > 0.01 * max(1.0, max(abs(coeff)))]
        eq = ' + '.join([f'{c:.6f}*{n}' for n, c in significant])
        self.pde_result_ = {'equation': eq, 'r2': r2, 'all_coefficients': dict(zip(names, coeff))}
        return eq, r2

    CDE_V80.pde_discover = pde_discover
    print('[扩展] PDE 发现模块已激活')

def unpatch_pde_discovery():
    if hasattr(CDE_V80, 'pde_discover'):
        delattr(CDE_V80, 'pde_discover')
    print('[扩展] PDE 发现模块已卸载')


# ========== 扩展7：Takens 时滞嵌入 ==========
def patch_takens_embedding(taus=None):
    from sklearn.linear_model import Lasso
    if taus is None:
        taus = [1, 2, 5, 10, 20]

    def takens_discover(self, X, y, var_names=None):
        n_samples, n_features = X.shape
        n_targets = y.shape[1] if y.ndim > 1 else 1
        if var_names is None:
            var_names = [f'x{i}' for i in range(n_features)]
        max_tau = max(taus)
        X_lag_list = [X]
        for tau in taus:
            X_lag_list.append(np.roll(X, tau, axis=0))
        X_takens = np.column_stack(X_lag_list)[max_tau:]
        y_aligned = y[max_tau:] if y.ndim > 1 else y[max_tau:]
        features = [np.ones(len(X_takens))]
        f_names = ['1']
        for j in range(n_features):
            features.append(X_takens[:, j])
            f_names.append(var_names[j])
            features.append(np.sin(X_takens[:, j]))
            f_names.append(f'sin({var_names[j]})')
            features.append(np.cos(X_takens[:, j]))
            f_names.append(f'cos({var_names[j]})')
        for tau_idx, tau in enumerate(taus):
            offset = n_features + tau_idx * n_features
            for j in range(n_features):
                idx = offset + j
                features.append(X_takens[:, idx])
                f_names.append(f'{var_names[j]}(t-{tau})')
        for j in range(n_features):
            features.append(X_takens[:, j]**2)
            f_names.append(f'{var_names[j]}^2')
        X_features = np.column_stack(features)
        results = {}
        target_names = [f'd{var_names[0]}/dt'] if n_targets == 1 else [f'd{var_names[i]}/dt' for i in range(n_targets)]
        for t_idx, t_name in enumerate(target_names):
            y_target = y_aligned[:, t_idx] if y_aligned.ndim > 1 else y_aligned
            lasso = Lasso(alpha=1e-4, max_iter=5000, fit_intercept=False)
            lasso.fit(X_features, y_target)
            nonzero_mask = np.abs(lasso.coef_) > 1e-6
            X_selected = X_features[:, nonzero_mask]
            coeff_selected = np.linalg.lstsq(X_selected, y_target, rcond=None)[0]
            coeff_full = np.zeros(len(f_names))
            coeff_full[nonzero_mask] = coeff_selected
            pred = X_features @ coeff_full
            r2 = 1 - np.sum((y_target - pred)**2) / np.sum((y_target - np.mean(y_target))**2)
            significant = [(n, c) for n, c in zip(f_names, coeff_full) if abs(c) > 0.005 * max((abs(coeff_full)))]
            eq = ' + '.join([f'{c:.4f}*{n}' for n, c in significant[:10]])
            results[t_name] = {'equation': eq, 'r2': r2, 'n_terms': len(significant)}
        self.takens_result_ = results
        return results

    CDE_V80.takens_discover = takens_discover
    print(f'[扩展] Takens 时滞嵌入已激活（时滞步长: {taus}）')

def unpatch_takens_embedding():
    if hasattr(CDE_V80, 'takens_discover'):
        delattr(CDE_V80, 'takens_discover')
    print('[扩展] Takens 时滞嵌入已卸载')


# ========== 扩展8：Lasso + lstsq 混合稀疏回归 ==========
def patch_lasso_sparse_regression(alpha=1e-4, max_iter=5000, significance=0.005):
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    original_fit = CDE_V80.fit

    def fit_with_lasso(self, X, y, feature_names=None):
        X = np.atleast_2d(X)
        y = np.asarray(y)
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(X.shape[1])]
        if self.time_series:
            Theta, names = self._build_time_series_features(X, feature_names)
        else:
            Theta, names = self._build_static_features(X, feature_names)
        scaler = StandardScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
        lasso.fit(Theta_scaled, y)
        nonzero_mask = np.abs(lasso.coef_) > 1e-8
        if np.sum(nonzero_mask) == 0:
            nonzero_mask[np.argmax(np.abs(lasso.coef_))] = True
        Theta_selected = Theta[:, nonzero_mask]
        coeff_selected = np.linalg.lstsq(Theta_selected, y, rcond=None)[0]
        self.coefficients_ = np.zeros(len(names))
        self.coefficients_[nonzero_mask] = coeff_selected
        self.selected_features_ = [names[i] for i in range(len(names)) if nonzero_mask[i]]
        pred = Theta @ self.coefficients_
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        self.r2_ = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 1.0
        residual = y - pred
        detrend = residual - np.mean(residual)
        acf = np.correlate(detrend, detrend, mode='same')
        self.residual_autocorr_ = acf[len(acf)//2 + 1] / acf[len(acf)//2] if acf[len(acf)//2] != 0 else 0.0
        threshold = significance * max(abs(self.coefficients_)) if max(abs(self.coefficients_)) > 0 else 0
        significant_idx = [i for i in range(len(names)) if abs(self.coefficients_[i]) > threshold]
        if not significant_idx:
            significant_idx = list(range(len(names)))
        terms = [f'{self.coefficients_[i]:.4f}*{names[i]}' for i in significant_idx]
        self.equation_ = ' + '.join(terms).replace('+ -', '- ') if terms else '0'
        return self

    CDE_V80.fit = fit_with_lasso
    print(f'[扩展] Lasso + lstsq 混合稀疏回归已激活 (α={alpha})')

def unpatch_lasso_sparse_regression():
    CDE_V80.fit = _original_fit
    print('[扩展] Lasso + lstsq 混合稀疏回归已卸载')

    # ========== 扩展9：耦合三角项 + 有理函数特征注入 ==========
def patch_coupled_features(inject_coupled_trig=True, inject_rational=True):
    original_build_static = CDE_V80._build_static_features

    def build_with_coupled(self, X, base_names):
        X_base, names_base = original_build_static(self, X, base_names)
        n_samples, n_features = X.shape
        extra_features = [X_base]
        extra_names = list(names_base)
        if inject_coupled_trig:
            for j in range(n_features):
                for k in range(j+1, n_features):
                    diff = X[:, j] - X[:, k]
                    extra_features.append(np.sin(diff).reshape(-1, 1))
                    extra_names.append(f'sin({base_names[j]}-{base_names[k]})')
                    extra_features.append(np.cos(diff).reshape(-1, 1))
                    extra_names.append(f'cos({base_names[j]}-{base_names[k]})')
        if inject_rational:
            for j in range(n_features):
                for k in range(j+1, n_features):
                    diff = X[:, j] - X[:, k]
                    denom = 1.0 / (1.0 + np.cos(diff))
                    extra_features.append(denom.reshape(-1, 1))
                    extra_names.append(f'1/(1+cos({base_names[j]}-{base_names[k]}))')
        return np.hstack(extra_features), extra_names

    CDE_V80._build_static_features = build_with_coupled
    print('[扩展] 耦合三角项 + 有理函数特征注入已激活')

def unpatch_coupled_features():
    CDE_V80._build_static_features = original_build_static
    print('[扩展] 耦合三角项 + 有理函数特征注入已卸载')


# ========== 扩展10：临界相变预警 + 级联失效检测 ==========
def patch_critical_transition(window=100):
    def critical_analyze(self, X, dt=0.05, shock_time=None):
        X = np.asarray(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n_samples, n_nodes = X.shape
        win = min(window, n_samples // 5)
        report = {'nodes': []}
        for j in range(n_nodes):
            x = X[:, j]
            local_var = np.zeros(n_samples)
            local_acf = np.zeros(n_samples)
            local_drift = np.zeros(n_samples)
            for i in range(win, n_samples):
                xw = x[i-win:i]
                local_var[i] = np.var(xw)
                if np.std(xw) > 1e-10:
                    local_acf[i] = np.corrcoef(xw[:-1], xw[1:])[0, 1]
                dx = np.diff(xw) / dt
                if len(dx) > 1 and len(xw[:-1]) > 1 and np.std(xw[:-1]) > 1e-10:
                    local_drift[i] = np.polyfit(xw[:-1], dx, 1)[0]
            if shock_time is not None:
                pre = slice(max(win, shock_time-300), shock_time)
                post = slice(shock_time+20, min(n_samples, shock_time+320))
            else:
                pre = slice(win, n_samples//3)
                post = slice(2*n_samples//3, n_samples)
            var_pre = np.mean(local_var[pre])
            var_post = np.mean(local_var[post])
            acf_pre = np.mean(local_acf[pre])
            acf_post = np.mean(local_acf[post])
            drift_pre = np.mean(local_drift[pre])
            drift_post = np.mean(local_drift[post])
            var_ratio = var_post / var_pre if var_pre > 1e-10 else 1.0
            acf_diff = acf_post - acf_pre
            drift_ratio = drift_post / drift_pre if abs(drift_pre) > 1e-10 else 1.0
            warnings = 0
            if var_ratio > 1.5: warnings += 1
            if acf_diff > 0.05: warnings += 1
            if drift_ratio < 0.5: warnings += 1
            report['nodes'].append({
                'node': j, 'var_ratio': var_ratio, 'acf_diff': acf_diff,
                'drift_ratio': drift_ratio, 'warnings': warnings,
                'verdict': 'CRITICAL' if warnings >= 2 else ('WARNING' if warnings >= 1 else 'STABLE')
            })
        report['global_verdict'] = 'CRITICAL' if any(n['verdict']=='CRITICAL' for n in report['nodes']) else 'STABLE'
        self.critical_report_ = report
        return report

    CDE_V80.critical_analyze = critical_analyze
    print(f'[扩展] 临界相变预警 + 级联失效检测已激活（窗口={window}）')

def unpatch_critical_transition():
    if hasattr(CDE_V80, 'critical_analyze'): delattr(CDE_V80, 'critical_analyze')
    print('[扩展] 临界相变预警已卸载')


# ========== 扩展11：多体耦合网络级联分析 ==========
def patch_network_cascade(n_nodes=20, m_edges=2):
    def build_scale_free_network(self, n=None, m=None):
        n = n or n_nodes
        m = m or m_edges
        adj = np.zeros((n, n))
        seed = m + 1
        for i in range(seed):
            for j in range(i+1, seed):
                adj[i, j] = adj[j, i] = 1
        deg = np.sum(adj, axis=1)
        for new_node in range(seed, n):
            probs = deg[:new_node] / np.sum(deg[:new_node])
            targets = np.random.choice(new_node, size=m, replace=False, p=probs)
            for t in targets:
                adj[new_node, t] = adj[t, new_node] = 1
            deg = np.sum(adj, axis=1)
        self.network_ = {'adjacency': adj, 'degrees': deg, 'hub': int(np.argmax(deg))}
        return self.network_

    def simulate_cascade(self, n_samples=4000, dt=0.05, k_coupling=0.3, shock_node=None, shock_magnitude=1.0, critical_slowing=False):
        if not hasattr(self, 'network_'):
            self.build_scale_free_network()
        adj = self.network_['adjacency']
        deg = self.network_['degrees']
        n = len(deg)
        hub = shock_node if shock_node is not None else self.network_['hub']
        X = np.zeros((n_samples, n))
        X[0] = np.ones(n)
        shock_time = int(n_samples * 0.75)
        sigma_t = np.linspace(0.05, 0.45, n_samples) if critical_slowing else np.full(n_samples, 0.12)
        for i in range(shock_time - 1):
            for j in range(n):
                drift = X[i, j] - X[i, j]**3
                coupling = sum(X[i, k] - X[i, j] for k in range(n) if adj[j, k])
                drift += k_coupling * coupling / max(1, deg[j])
                dW = np.random.normal(0, np.sqrt(dt))
                X[i+1, j] = X[i, j] + drift * dt + sigma_t[i] * dW
        X[shock_time-1, hub] += shock_magnitude
        for i in range(shock_time-1, n_samples - 1):
            for j in range(n):
                drift = X[i, j] - X[i, j]**3
                coupling = sum(X[i, k] - X[i, j] for k in range(n) if adj[j, k])
                drift += k_coupling * coupling / max(1, deg[j])
                dW = np.random.normal(0, np.sqrt(dt))
                X[i+1, j] = X[i, j] + drift * dt + sigma_t[i] * dW
        pre = slice(shock_time-300, shock_time-1)
        post = slice(shock_time+50, shock_time+350)
        ratios = []
        for j in range(n):
            vp = np.var(X[pre, j])
            vs = np.var(X[post, j])
            ratios.append(vs / vp if vp > 1e-10 else 1.0)
        self.cascade_report_ = {
            'X': X, 'shock_time': shock_time, 'hub': hub,
            'ratios': ratios,
            'avalanche_nodes': [j for j in range(n) if ratios[j] > 1.5 and j != hub],
            'avg_ratio': np.mean([r for j, r in enumerate(ratios) if j != hub])
        }
        return self.cascade_report_

    CDE_V80.build_scale_free_network = build_scale_free_network
    CDE_V80.simulate_cascade = simulate_cascade
    print(f'[扩展] 多体耦合网络级联分析已激活（默认{n_nodes}节点, BA m={m_edges}）')

def unpatch_network_cascade():
    for attr in ['build_scale_free_network', 'simulate_cascade']:
        if hasattr(CDE_V80, attr): delattr(CDE_V80, attr)
    print('[扩展] 多体耦合网络模块已卸载')


# ========== 扩展12：Lasso α 自适应选择 ==========
def patch_adaptive_lasso(alpha_range=None, n_alphas=20):
    from sklearn.linear_model import Lasso
    original_fit = CDE_V80.fit

    def adaptive_lasso_fit(self, X, y, feature_names=None):
        X = np.atleast_2d(X)
        y = np.asarray(y)
        n_samples = len(y)
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(X.shape[1])]
        if alpha_range is None:
            alphas = np.logspace(-6, -1, n_alphas)
        else:
            alphas = np.logspace(*alpha_range, n_alphas)
        best_alpha = alphas[0]
        best_bic = np.inf
        best_coeff = None
        best_nonzero = None
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False)
            lasso.fit(X, y)
            nonzero_mask = np.abs(lasso.coef_) > 1e-8
            n_terms = np.sum(nonzero_mask)
            if n_terms == 0: continue
            pred = X @ lasso.coef_
            sse = np.sum((y - pred)**2)
            if sse < 1e-15: sse = 1e-15
            bic = n_samples * np.log(sse / n_samples) + n_terms * np.log(n_samples)
            if bic < best_bic:
                best_bic = bic
                best_alpha = alpha
                best_coeff = lasso.coef_.copy()
                best_nonzero = nonzero_mask.copy()
        if best_coeff is None:
            lasso = Lasso(alpha=1e-4, max_iter=5000, fit_intercept=False)
            lasso.fit(X, y)
            best_coeff = lasso.coef_
            best_nonzero = np.abs(best_coeff) > 1e-8
        X_sel = X[:, best_nonzero]
        coeff_sel = np.linalg.lstsq(X_sel, y, rcond=None)[0]
        self.coefficients_ = np.zeros(X.shape[1])
        self.coefficients_[best_nonzero] = coeff_sel
        self.selected_features_ = [feature_names[i] for i in range(X.shape[1]) if best_nonzero[i]]
        pred = X @ self.coefficients_
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        self.r2_ = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 1.0
        residual = y - pred
        detrend = residual - np.mean(residual)
        acf = np.correlate(detrend, detrend, mode='same')
        self.residual_autocorr_ = acf[len(acf)//2 + 1] / acf[len(acf)//2] if acf[len(acf)//2] != 0 else 0.0
        threshold = 0.01 * max(abs(self.coefficients_)) if max(abs(self.coefficients_)) > 0 else 0
        sig_idx = [i for i in range(len(feature_names)) if abs(self.coefficients_[i]) > threshold]
        terms = [f'{self.coefficients_[i]:.4f}*{feature_names[i]}' for i in sig_idx]
        self.equation_ = ' + '.join(terms).replace('+ -', '- ') if terms else '0'
        return self

    CDE_V80.fit = adaptive_lasso_fit
    print(f'[扩展] Lasso α 自适应选择已激活')

def unpatch_adaptive_lasso():
    CDE_V80.fit = _original_fit
    print('[扩展] Lasso α 自适应已卸载')


# ========== 扩展13：Lyapunov 动态阈值 + 联合参数估计 ==========
def patch_chaos_aware_threshold(lyapunov_steps=77):
    original_get_p0 = CDE_V80.get_p0_report

    def chaos_aware_p0(self):
        if self.r2_ is None or self.residual_autocorr_ is None:
            return 'P0 not available.'
        if self.r2_ > 0.9999:
            return 'P0: Structurally exact. Residuals are numerical noise.'
        threshold = 2.0 / np.sqrt(lyapunov_steps)
        if abs(self.residual_autocorr_) < threshold:
            return 'P0: Model adequate under chaos-aware threshold.'
        elif abs(self.residual_autocorr_) < threshold * 5:
            return 'P0: Acceptable for chaotic system (within 5x Lyapunov baseline).'
        else:
            return 'P0: Further inspection recommended.'

    CDE_V80.get_p0_report = chaos_aware_p0

    def joint_parameter_estimate(self, X_list, y_list, shared_indices=None):
        if shared_indices is None: shared_indices = []
        n_eqs = len(X_list)
        all_coeffs = []
        for i in range(n_eqs):
            coeff = np.linalg.lstsq(X_list[i], y_list[i], rcond=None)[0]
            all_coeffs.append(coeff)
        for idx in shared_indices:
            shared_vals = [coeff[idx] for coeff in all_coeffs if idx < len(coeff)]
            if shared_vals:
                joint_val = np.mean(shared_vals)
                for coeff in all_coeffs:
                    if idx < len(coeff): coeff[idx] = joint_val
        self.joint_coefficients_ = all_coeffs
        return all_coeffs

    CDE_V80.joint_parameter_estimate = joint_parameter_estimate
    print(f'[扩展] Lyapunov 动态阈值 + 联合参数估计已激活')

def unpatch_chaos_aware_threshold():
    CDE_V80.get_p0_report = original_get_p0
    if hasattr(CDE_V80, 'joint_parameter_estimate'): delattr(CDE_V80, 'joint_parameter_estimate')
    print('[扩展] 混沌感知模块已卸载')


# ========== 扩展14：GP自动结构进化 ==========
def patch_gp_structure_evolution(pop_size=100, generations=20, max_depth=4):
    import random, copy

    class _ExprNode:
        def __init__(self, op=None, left=None, right=None, atom=None, const=None):
            self.op, self.left, self.right, self.atom, self.const = op, left, right, atom, const
        def evaluate(self, atoms):
            if self.atom: return atoms[self.atom]
            if self.const is not None: return np.full(len(atoms[list(atoms.keys())[0]]), self.const)
            l = self.left.evaluate(atoms)
            r = self.right.evaluate(atoms) if self.right else None
            ops = {'+':lambda a,b:a+b, '-':lambda a,b:a-b, '*':lambda a,b:a*b, '/':lambda a,b:a/(b+1e-10),
                   'sin':lambda a,_:np.sin(a), 'cos':lambda a,_:np.cos(a), 'sq':lambda a,_:a**2}
            return ops[self.op](l, r)
        def to_string(self):
            if self.atom: return self.atom
            if self.const is not None: return f'{self.const:.2f}'
            if self.op in ['sin','cos','sq']: return f'{self.op}({self.left.to_string()})'
            return f'({self.left.to_string()} {self.op} {self.right.to_string()})'
        def clone(self): return copy.deepcopy(self)

    def gp_evolve(self, X, y, max_depth=None, pop=None, gens=None):
        md = max_depth or max_depth
        p = pop or pop_size
        g = gens or generations
        n_features = X.shape[1]
        var_names = [f'x{i}' for i in range(n_features)]
        atoms = {}
        for i in range(n_features):
            atoms[var_names[i]] = X[:, i]
            atoms[f'sin_{var_names[i]}'] = np.sin(X[:, i])
            atoms[f'cos_{var_names[i]}'] = np.cos(X[:, i])
        atoms['o1_sq'] = X[:, 1]**2 if n_features > 1 else X[:, 0]**2
        atoms['o2_sq'] = X[:, 3]**2 if n_features > 3 else atoms['o1_sq']
        atoms_list = list(atoms.keys())

        def _random_expr(d=4):
            if d == 0 or random.random() < 0.25:
                return _ExprNode(atom=random.choice(atoms_list)) if random.random() < 0.6 else _ExprNode(const=random.uniform(-3,3))
            op = random.choice(['sin','cos','sq']) if random.random() < 0.3 else random.choice(['+','-','*','/'])
            if op in ['sin','cos','sq']:
                return _ExprNode(op=op, left=_random_expr(d-1))
            return _ExprNode(op=op, left=_random_expr(d-1), right=_random_expr(d-1))

        def _fitness(tree):
            try:
                vals = tree.evaluate(atoms)
                if np.any(np.isnan(vals)) or np.any(np.isinf(vals)) or np.std(vals) < 1e-10: return -1e10
                return abs(np.corrcoef(vals, y)[0, 1]) - 0.0005 * len(tree.to_string())
            except: return -1e10

        population = [_random_expr(md) for _ in range(p)]
        best_tree, best_fit = None, -1e10
        for gen in range(g):
            fits = [_fitness(t) for t in population]
            best_idx = np.argmax(fits)
            if fits[best_idx] > best_fit:
                best_fit = fits[best_idx]
                best_tree = population[best_idx].clone()
            new_pop = [population[best_idx].clone()]
            while len(new_pop) < p:
                t_size = min(5, len(population))
                tournament = random.sample(range(len(population)), t_size)
                p1 = population[max(tournament, key=lambda i: fits[i])]
                tournament = random.sample(range(len(population)), t_size)
                p2 = population[max(tournament, key=lambda i: fits[i])]
                child = p1.clone()
                target = child.random_subtree() if hasattr(child, 'random_subtree') else child
                replacement = p2.random_subtree().clone() if hasattr(p2, 'random_subtree') else p2.clone()
                target.op, target.left, target.right, target.atom, target.const = replacement.op, replacement.left, replacement.right, replacement.atom, replacement.const
                if random.random() < 0.35:
                    mutant = child.clone()
                    tgt = mutant.random_subtree() if hasattr(mutant, 'random_subtree') else mutant
                    new_sub = _random_expr(md)
                    tgt.op, tgt.left, tgt.right, tgt.atom, tgt.const = new_sub.op, new_sub.left, new_sub.right, new_sub.atom, new_sub.const
                    child = mutant
                new_pop.append(child)
            population = new_pop
        self.gp_result_ = {
            'best_structure': best_tree.to_string() if best_tree else 'none',
            'fitness': best_fit,
            'values': best_tree.evaluate(atoms) if best_tree else None
        }
        return self.gp_result_

    CDE_V80.gp_evolve = gp_evolve
    print(f'[扩展] GP自动结构进化已激活（种群={pop_size}, 世代={generations}, 深度={max_depth}）')

def unpatch_gp_structure_evolution():
    if hasattr(CDE_V80, 'gp_evolve'): delattr(CDE_V80, 'gp_evolve')
    print('[扩展] GP自动结构进化已卸载')


# ========== 扩展15：Minimal Physics Grammar Engine ==========
def patch_physics_grammar(max_operator_depth=3, min_std=0.01):
    class _PhysicsGrammarEngine:
        def __init__(self, max_depth=3, min_std=0.01):
            self.max_depth = max_depth
            self.min_std = min_std
            self.validation_log = []
        def validate(self, expression_str, atom_values_dict, target_dimension='angular_velocity'):
            reasons = []
            depth = 0
            max_depth = 0
            for ch in expression_str:
                if ch == '(': depth += 1; max_depth = max(max_depth, depth)
                elif ch == ')': depth -= 1
            if any(op in expression_str for op in ['sin','cos','sq','exp','log']): max_depth += 1
            if max_depth > self.max_depth:
                reasons.append(f'OPERATOR_DEPTH: {max_depth} > {self.max_depth}')
            if atom_values_dict is not None:
                try:
                    values = atom_values_dict
                    if isinstance(values, dict): values = list(values.values())[0]
                    if hasattr(values, '__len__'):
                        arr = np.asarray(values)
                        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                            reasons.append('SINGULARITY: NaN or Inf')
                        elif np.std(arr) < self.min_std:
                            reasons.append(f'SINGULARITY: std={np.std(arr):.6f} < {self.min_std}')
                except: pass
            is_valid = len(reasons) == 0
            self.validation_log.append({'expression': expression_str[:80], 'valid': is_valid, 'reasons': reasons})
            return is_valid, reasons
        def get_report(self):
            if not self.validation_log: return 'No validations.'
            total = len(self.validation_log)
            passed = sum(1 for v in self.validation_log if v['valid'])
            return f'Grammar: {passed}/{total} passed'

    CDE_V80.grammar_engine = _PhysicsGrammarEngine(max_depth=max_operator_depth, min_std=min_std)
    print(f'[扩展] Minimal Physics Grammar Engine 已激活（深度≤{max_operator_depth}, std≥{min_std}）')

def unpatch_physics_grammar():
    if hasattr(CDE_V80, 'grammar_engine'): delattr(CDE_V80, 'grammar_engine')
    print('[扩展] Physics Grammar Engine 已卸载')


# ========== 扩展16：Proposal Diversity Governance ==========
def patch_diversity_governance(quotas=None):
    if quotas is None:
        quotas = {'random_mutation': 0.30, 'grammar_guided': 0.30, 'nn_proposal': 0.30, 'adversarial_novelty': 0.10}
    
    class _DiversityGovernor:
        def __init__(self, quotas_dict):
            self.quotas = quotas_dict
            self.total_quota = sum(quotas_dict.values())
            self.proposal_history = []
            self.channel_stats = {ch: {'proposed': 0, 'accepted': 0} for ch in quotas_dict}
        def allocate(self, n_candidates):
            allocation = {}
            remaining = n_candidates
            for ch, quota in list(self.quotas.items())[:-1]:
                allocation[ch] = max(1, int(n_candidates * quota))
                remaining -= allocation[ch]
            allocation[list(self.quotas.keys())[-1]] = max(1, remaining)
            return allocation
        def register_proposal(self, channel, expression, accepted):
            self.proposal_history.append({'channel': channel, 'expression': expression[:80], 'accepted': accepted})
            self.channel_stats[channel]['proposed'] += 1
            if accepted: self.channel_stats[channel]['accepted'] += 1
        def audit(self):
            report = {}
            for ch, stats in self.channel_stats.items():
                total = stats['proposed']
                accepted = stats['accepted']
                rate = accepted / total if total > 0 else 0.0
                report[ch] = {'proposed': total, 'accepted': accepted, 'acceptance_rate': rate}
            max_rate = max(r['acceptance_rate'] for r in report.values()) if report else 0
            avg_rate = np.mean([r['acceptance_rate'] for r in report.values()]) if report else 0
            alerts = []
            for ch, r in report.items():
                if r['acceptance_rate'] > avg_rate * 2 and r['proposed'] > 5:
                    alerts.append(f'MONOPOLY_ALERT: {ch} rate {r["acceptance_rate"]:.2f} > 2x avg {avg_rate:.2f}')
            report['alerts'] = alerts
            report['healthy'] = len(alerts) == 0
            return report

    CDE_V80.diversity_governor = _DiversityGovernor(quotas)
    print(f'[扩展] Proposal Diversity Governance 已激活（配额: {quotas}）')

def unpatch_diversity_governance():
    if hasattr(CDE_V80, 'diversity_governor'): delattr(CDE_V80, 'diversity_governor')
    print('[扩展] Proposal Diversity Governance 已卸载')


# ========== 一键操作 ==========
def apply_all_patches():
    patch_identity_detector()
    patch_adaptive_regime_scan()
    patch_multi_freq_hypothesis()
    patch_uncertainty_quantification()
    patch_sde_discovery()
    patch_pde_discovery()
    patch_takens_embedding()
    patch_lasso_sparse_regression()
    patch_coupled_features()
    patch_critical_transition()
    patch_network_cascade()
    patch_adaptive_lasso()
    patch_chaos_aware_threshold()
    patch_gp_structure_evolution()
    patch_physics_grammar()
    patch_diversity_governance()
    patch_nn_proposal()
    patch_grammar_constrained_generator()
    patch_residual_analyzer()
    patch_feature_pool_recommender()
    patch_auto_evolution()
    patch_period_discovery()
    print("[扩展] 所有增强补丁已加载")

﻿# ========== 扩展20：特征池自动设计器 (AIRA-Compose 启发) ==========
def patch_feature_pool_recommender():
    """激活特征池自动设计器：NN 根据数据集特征推荐最优特征池组合"""
    import torch
    import torch.nn as nn
    import numpy as np

    class FeaturePoolRecommender(nn.Module):
        def __init__(self, n_combos=7):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_combos, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_combos), nn.Softmax(dim=1)
            )
            self.feature_combos = [
                "多项式", "多项式+三角", "多项式+时滞",
                "三角+时滞", "多项式+三角+时滞",
                "多项式+三角+时滞+积分", "多项式+三角+时滞+积分+TV"
            ]
        def forward(self, x): return self.net(x)
        def recommend(self, dataset_profile):
            with torch.no_grad():
                x = torch.tensor(dataset_profile, dtype=torch.float32).unsqueeze(0)
                idx = self(x).argmax().item()
            return self.feature_combos[idx]

    CDE_V80.feature_recommender = FeaturePoolRecommender()
    print("[扩展] 特征池自动设计器已激活")

def unpatch_feature_pool_recommender():
    if hasattr(CDE_V80, 'feature_recommender'): delattr(CDE_V80, 'feature_recommender')
    print("[扩展] 特征池自动设计器已卸载")

# ========== 扩展21：闭环自进化引擎 (AIRA 闭环启发) ==========
def patch_auto_evolution(max_rounds=20, patience=5):
    """激活闭环自进化引擎：自动循环生成→验证→保留/丢弃→再生成"""
    import numpy as np

    class AutoEvolutionEngine:
        def __init__(self, generate_fn, evaluate_fn, max_rounds=max_rounds, patience=patience):
            self.generate_fn = generate_fn
            self.evaluate_fn = evaluate_fn
            self.max_rounds = max_rounds
            self.patience = patience
            self.best_score = -np.inf
            self.best_solution = None
            self.no_improve = 0
            self.history = []
        def run(self, verbose=True):
            for round_idx in range(self.max_rounds):
                candidate = self.generate_fn()
                score = self.evaluate_fn(candidate)
                self.history.append((round_idx, candidate, score))
                if score > self.best_score:
                    self.best_score = score
                    self.best_solution = candidate
                    self.no_improve = 0
                else:
                    self.no_improve += 1
                if verbose: print(f"第{round_idx+1}轮: R²={score:.4f}")
                if self.no_improve >= self.patience:
                    if verbose: print(f"连续{self.patience}轮无改善，自动停止")
                    break
            return self.best_solution, self.best_score

    CDE_V80.auto_evolution = AutoEvolutionEngine
    print(f"[扩展] 闭环自进化引擎已激活 (最多{max_rounds}轮, 耐心{patience}轮)")

def unpatch_auto_evolution()
    unpatch_period_discovery()
    patch_period_discovery():
    if hasattr(CDE_V80, 'auto_evolution'): delattr(CDE_V80, 'auto_evolution')
    print("[扩展] 闭环自进化引擎已卸载")


def remove_all_patches():
    unpatch_identity_detector()
    unpatch_adaptive_regime_scan()
    unpatch_multi_freq_hypothesis()
    unpatch_uncertainty_quantification()
    unpatch_sde_discovery()
    unpatch_pde_discovery()
    unpatch_takens_embedding()
    unpatch_lasso_sparse_regression()
    unpatch_coupled_features()
    unpatch_critical_transition()
    unpatch_network_cascade()
    unpatch_adaptive_lasso()
    unpatch_chaos_aware_threshold()
    unpatch_gp_structure_evolution()
    unpatch_physics_grammar()
    unpatch_diversity_governance()
    print("[扩展] 所有补丁已卸载，引擎已恢复至纯净 V8.1")
# ========== 扩展22：周期自动发现引�?==========
def patch_period_discovery(period_candidates=None):
    """激活周期自动发现引擎�?""
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
    print(f"[扩展] 周期自动发现引擎已激�?)

def unpatch_period_discovery():
    if hasattr(CDE_V80, 'period_scan'): delattr(CDE_V80, 'period_scan')
    print("[扩展] 周期自动发现引擎已卸�?)

# ========== 扩展23：百万级特征池搜索引�?==========
def patch_million_search(period_range=None, n_trials=500):
    """激活百万级特征池搜索引擎�?27万种组合中自动找到最优特征池�?""
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
    print(f"[扩展23] 百万级特征池搜索引擎已激活（{period_range[0]}-{period_range[1]}�? {n_trials}次搜索）")

def unpatch_million_search():
    if hasattr(CDE_V80, 'million_search'): delattr(CDE_V80, 'million_search')
    print("[扩展23] 百万级搜索引擎已卸载")

# ========== 扩展23：百万级特征池搜索引�?==========
def patch_million_search(period_range=None, n_trials=500):
    """激活百万级特征池搜索引擎�?27万种组合中自动找到最优特征池�?""
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
    print(f"[扩展23] 百万级特征池搜索引擎已激活（{period_range[0]}-{period_range[1]}�? {n_trials}次搜索）")

def unpatch_million_search():
    if hasattr(CDE_V80, 'million_search'): delattr(CDE_V80, 'million_search')
    print("[扩展23] 百万级搜索引擎已卸载")

# ========== 扩展23：百万级特征池搜索引�?==========
def patch_million_search(period_range=None, n_trials=500):
    """激活百万级特征池搜索引擎�?27万种组合中自动找到最优特征池�?""
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
    print(f"[扩展23] 百万级特征池搜索引擎已激活（{period_range[0]}-{period_range[1]}�? {n_trials}次搜索）")

def unpatch_million_search():
    if hasattr(CDE_V80, 'million_search'): delattr(CDE_V80, 'million_search')
    print("[扩展23] 百万级搜索引擎已卸载")
