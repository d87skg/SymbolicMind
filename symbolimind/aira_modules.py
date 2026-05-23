
"""
扩展20：特征池自动设计器 (AIRA-Compose 启发)
扩展21：闭环自进化引擎 (AIRA 闭环启发)
"""

import numpy as np
import torch
import torch.nn as nn

# ========== 扩展20：特征池自动设计器 ==========
class FeaturePoolRecommender(nn.Module):
    """根据数据集特征自动推荐最优特征池组合"""
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
        """输入数据集特征向量，返回推荐的特征池"""
        with torch.no_grad():
            x = torch.tensor(dataset_profile, dtype=torch.float32).unsqueeze(0)
            idx = self(x).argmax().item()
        return self.feature_combos[idx]

# ========== 扩展21：闭环自进化引擎 ==========
class AutoEvolutionEngine:
    """自动循环：生成→验证→保留/丢弃→再生成"""
    def __init__(self, generate_fn, evaluate_fn, max_rounds=20, patience=5):
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
            
            if verbose:
                status = "✅" if score > self.best_score else "⏸️"
                print(f"第{round_idx+1}轮: R²={score:.4f} {status}")
            
            if self.no_improve >= self.patience:
                if verbose: print(f"⏹️ 连续{self.patience}轮无改善，自动停止")
                break
        
        return self.best_solution, self.best_score

print("✅ AIRA 启发模块代码已生成")
print("请将以上代码保存到 symbolimind/extensions.py")
