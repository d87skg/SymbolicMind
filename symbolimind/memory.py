"""
SymbolicMind Memory - Local Experience Buffer

Stores successful discoveries and allows instant recall.
Supports on/off switch for memory mode.

Author: [Your Name]
License: MIT
"""
import os, json, hashlib, numpy as np
from datetime import datetime

class CDEMemory:
    """CDE 本地经验库：可开关，越用越聪明。"""
    def __init__(self, memory_file=os.path.join(os.path.dirname(__file__), "..", "memory", "cde_experience.json"), enabled=True):
        self.memory_file = memory_file
        self.enabled = enabled
        self.memory = {}
        if self.enabled:
            self._load()

    def _load(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            self.memory = {}

    def _save(self):
        if self.enabled:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def memorize(self, data_csv, target_column, result):
        """记住一次成功的发现。"""
        if not self.enabled or result.get("r2", 0) < 0.95:
            return
        # 生成数据指纹
        fingerprint = hashlib.md5(data_csv.encode()).hexdigest()
        # 提取关键信息
        entry = {
            "equation": result["equation"],
            "r2": result["r2"],
            "p0": result["p0_verdict"],
            "target": target_column,
            "timestamp": datetime.now().isoformat(),
            "sample_columns": list(np.genfromtxt([data_csv], delimiter=',', names=True, max_rows=1).dtype.names)
        }
        self.memory[fingerprint] = entry
        self._save()
        return fingerprint

    def recall(self, data_csv):
        """尝试回忆类似数据的成功经验。"""
        if not self.enabled:
            return None
        fingerprint = hashlib.md5(data_csv.encode()).hexdigest()
        return self.memory.get(fingerprint)
