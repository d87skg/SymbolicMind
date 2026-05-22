"""
SymbolicMind Engine - Core Symbolic Regression Module

This module provides the CDE_V80 class, the core engine for discovering
governing equations from data. It features BIC-guided sparse regression,
P0 falsifiability boundary, and dual-mode operation (static / time-series).

Author: [Your Name]
License: MIT
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class CDE_V80:
    """
    CDE V8.0 符号回归适配器 (SRBench 兼容)
    
    双模式：
    - time_series=False (默认): 静态表格，使用多项式+超越函数特征池
    - time_series=True: 时间序列，启用时滞、积分核等完整记忆算子
    """
    def __init__(self, threshold=0.2, max_iter=8, time_series=False, autostop=True):
        self.threshold = threshold
        self.max_iter = max_iter
        self.time_series = time_series
        self.autostop = autostop
        self.coefficients_ = None
        self.selected_features_ = None
        self.r2_ = None
        self.residual_autocorr_ = None
        self.equation_ = None

    def fit(self, X, y, feature_names=None):
        X = np.atleast_2d(X)
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(X.shape[1])]
        
        # 特征工程
        if self.time_series:
            X_aug, aug_names = self._build_time_series_features(X, feature_names)
        else:
            X_aug, aug_names = self._build_static_features(X, feature_names)
        
        # 核心 BIC 精拟合
        final_coeffs, active_idx, r2, lag1 = self._bic_fit(X_aug, y, aug_names)
        
        self.coefficients_ = final_coeffs
        self.selected_features_ = [aug_names[i] for i in active_idx]
        self.r2_ = r2
        self.residual_autocorr_ = lag1
        self.equation_ = self._format_equation(final_coeffs, aug_names, active_idx)
        
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if self.time_series:
            X_aug, _ = self._build_time_series_features(X, [f'x{i}' for i in range(X.shape[1])])
        else:
            X_aug, _ = self._build_static_features(X, [f'x{i}' for i in range(X.shape[1])])
        return X_aug @ self.coefficients_

    def get_equation(self):
        if self.equation_ is None:
            return "Model not fitted."
        return self.equation_

    def get_p0_report(self):
        if self.residual_autocorr_ is None:
            return "P0 not available."
        if self.r2_ is None:
            return "P0 not available."
        if self.r2_ < 0.3 and abs(self.residual_autocorr_) > 0.3:
            return "Hidden variables detected (low R^2 + high residual autocorrelation)"
        elif self.r2_ >= 0.95 and abs(self.residual_autocorr_) < 0.1:
            return "Model is structurally adequate."
        else:
            return "Further inspection recommended."

    def _build_static_features(self, X, base_names):
        """静态模式：多项式 + 超越函数"""
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X)
        poly_names = poly.get_feature_names_out(base_names)
        
        # 添加超越函数特征
        trans_list = [X_poly]
        trans_names = list(poly_names)
        
        n_samples, n_features = X.shape
        for i in range(n_features):
            xi = X[:, i]
            # 安全处理，防止除零/溢出
            trans_list.append(np.sqrt(np.abs(xi)).reshape(-1, 1))
            trans_names.append(f'sqrt_{base_names[i]}')
            trans_list.append(np.exp(-xi**2).reshape(-1, 1))
            trans_names.append(f'exp_neg_{base_names[i]}_sq')
            trans_list.append(np.sin(xi).reshape(-1, 1))
            trans_names.append(f'sin_{base_names[i]}')
            trans_list.append(np.cos(xi).reshape(-1, 1))
            trans_names.append(f'cos_{base_names[i]}')
            trans_list.append(np.log1p(np.abs(xi)).reshape(-1, 1))
            trans_names.append(f'log1p_{base_names[i]}')
            # 反比例项
            trans_list.append((1.0 / (1.0 + np.abs(xi))).reshape(-1, 1))
            trans_names.append(f'inv1p_{base_names[i]}')
        
        return np.hstack(trans_list), trans_names

    def _build_time_series_features(self, X, base_names):
        """时间序列模式：基础项 + 时滞 + 积分核"""
        n_samples, n_features = X.shape
        dt = 1.0  # 假设均匀采样间隔为1
        
        feature_list = [X]
        names = list(base_names)
        
        # 时滞项
        for lag in [1, 2, 5, 10]:
            if lag < n_samples:
                for j in range(n_features):
                    rolled = np.roll(X[:, j], lag)
                    rolled[:lag] = X[0, j]  # 边界填充
                    feature_list.append(rolled.reshape(-1, 1))
                    names.append(f'{base_names[j]}_lag{lag}')
        
        # 连续指数积分核 (多个衰减率)
        alphas = [0.05, 0.1, 0.2, 0.5]
        for j in range(n_features):
            for alpha in alphas:
                mem = np.zeros(n_samples)
                cum_sum = 0.0
                for i in range(n_samples):
                    if i == 0:
                        cum_sum = X[0, j] * dt
                    else:
                        cum_sum = np.exp(-alpha * dt) * cum_sum + X[i, j] * dt
                    mem[i] = cum_sum
                feature_list.append(mem.reshape(-1, 1))
                names.append(f'int_{base_names[j]}_a{alpha}')
        
        # 时变核 (示例：正弦调制)
        t = np.arange(n_samples) * dt
        for j in range(n_features):
            for amp in [0.1, 0.2]:
                alpha_t = 0.1 + amp * np.sin(2 * np.pi * t / 25.0)
                mem = np.zeros(n_samples)
                for i in range(n_samples):
                    integral = 0.0
                    for k in range(i):
                        tau = (i - k) * dt
                        integral += np.exp(-alpha_t[i] * tau) * X[k, j] * dt
                    mem[i] = integral
                feature_list.append(mem.reshape(-1, 1))
                names.append(f'tv_int_{base_names[j]}_amp{amp}')
        
        return np.hstack(feature_list), names

    def _bic_fit(self, Theta, target, feature_names):
        """扁平两阶段 + BIC 后向剔除"""
        Theta_std = np.std(Theta, axis=0)
        Theta_std[Theta_std == 0] = 1.0
        Theta_scaled = Theta / Theta_std
        n_samples, n_features = Theta.shape
        active = list(range(n_features))

        def compute_bic(pred, target, k):
            sse = np.sum((target - pred)**2)
            if sse < 1e-15: sse = 1e-15
            return n_samples * np.log(sse / n_samples) + k * np.log(n_samples)

        # 初始全模型拟合
        coeffs_full = np.linalg.lstsq(Theta_scaled, target, rcond=1e-6)[0]
        pred_full = Theta_scaled @ coeffs_full
        best_bic = compute_bic(pred_full, target, len(active))

        # 后向剔除
        while len(active) > 1:
            candidate_bics = []
            for col in active:
                trial = [c for c in active if c != col]
                Theta_trial = Theta_scaled[:, trial]
                coeff_trial = np.linalg.lstsq(Theta_trial, target, rcond=1e-6)[0]
                pred_trial = Theta_trial @ coeff_trial
                bic_trial = compute_bic(pred_trial, target, len(trial))
                candidate_bics.append((bic_trial, col))
            candidate_bics.sort(key=lambda v: v[0])
            min_bic, drop_col = candidate_bics[0]
            if min_bic < best_bic:
                active.remove(drop_col)
                best_bic = min_bic
            else:
                break

        # 最终系数还原到原始尺度
        final_coeffs = np.zeros(n_features)
        if len(active) > 0:
            Theta_active_scaled = Theta_scaled[:, active]
            coeff_active = np.linalg.lstsq(Theta_active_scaled, target, rcond=1e-6)[0]
            final_coeffs[active] = coeff_active / Theta_std[active]

        # R² 和残差自相关
        pred = Theta @ final_coeffs
        ss_res = np.sum((target - pred)**2)
        ss_tot = np.sum((target - np.mean(target))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 1.0

        residual = target - pred
        residual_detrend = residual - np.mean(residual)
        acf = np.correlate(residual_detrend, residual_detrend, mode='same')
        lag1 = acf[len(acf)//2 + 1] / acf[len(acf)//2] if acf[len(acf)//2] != 0 else 0.0

        return final_coeffs, active, r2, lag1

    def _format_equation(self, coeffs, names, active_idx):
        """将系数格式化为方程字符串"""
        if len(active_idx) == 0:
            return "y = 0 (all terms eliminated)"
        terms = []
        for i in active_idx:
            c = coeffs[i]
            if abs(c) > 1e-10:
                terms.append(f"{round(c, 6)}*{names[i]}")
        if not terms:
            return "y = constant"
        return "y = " + " + ".join(terms).replace("+ -", "- ")


# 兼容旧版别名
CDE_V6_6 = CDE_V80
