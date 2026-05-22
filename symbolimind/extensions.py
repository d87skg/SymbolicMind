"""
SymbolicMind Extensions - Optional Patches

Provides identity detector and adaptive regime scanner.
Can be loaded on-demand via apply_all_patches().

Author: [Your Name]
License: MIT
"""
import numpy as np
from symbolimind.engine import CDE_V80

# ---------- 原始备份 ----------
_original_fit = CDE_V80.fit

# ---------- 升级版恒等式检测 ----------
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

# ---------- 自适应断层扫描 ----------
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

# ---------- 一键操作 ----------
def apply_all_patches():
    patch_identity_detector()
    patch_adaptive_regime_scan()
    print("[扩展] 所有增强补丁已加载")

def remove_all_patches():
    unpatch_identity_detector()
    unpatch_adaptive_regime_scan()
    print("[扩展] 所有补丁已卸载，引擎恢复至纯净 V8.1")
