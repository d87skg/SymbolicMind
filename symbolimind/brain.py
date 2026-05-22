"""
SymbolicMind Brain - Intelligent Agent Frontend

Contains LocalBrain and GPTBrain for natural language interaction
with the CDE engine. Includes physical knowledge base and equation beautification.

Author: [Your Name]
License: MIT
"""
import os, sys, json
import numpy as np

class LocalBrain:
    def __init__(self):
        # ===== 内置物理知识库 =====
        self.physical_terms = {
            "加速度": "d2x_dt2",
            "速度": "dx_dt",
            "位置": "x",
            "位移": "x",
            "二阶导数": "d2x_dt2",
            "一阶导数": "dx_dt"
        }

    def parse_question(self, question, available_columns):
        """从自然语言中提取目标列名，优先使用物理知识库。"""
        q_lower = question.lower()
        
        # 1. 优先查找物理概念
        for term, col in self.physical_terms.items():
            if term in question and col in available_columns:
                return col
        
        # 2. 直接匹配列名（按长度降序）
        sorted_cols = sorted(available_columns, key=lambda c: len(c), reverse=True)
        for col in sorted_cols:
            if col in q_lower:
                return col
        
        # 3. 默认返回最后一个
        return available_columns[-1]

    def generate_answer(self, result, target_column=None):
        """生成专业的科学报告，并自动将方程重写为物理形式。"""
        if "error" in result:
            return f"抱歉，分析遇到问题：{result['error']}"
        
        equation = result['equation']
        r2 = result['r2']
        p0 = result['p0_verdict']
        lac = result.get('residual_autocorr', 0.0)
        
        # 将方程重写为更物理的形式
        phys_eq = self._format_equation_physically(equation, target_column)
        
        # P0 解读
        if "adequate" in p0.lower() or "exact" in p0.lower():
            reliability = "模型通过了严格的P0可证伪性检验，方程结构可靠。"
        else:
            reliability = f"模型拟合优度很高，但P0检测发现残差仍有自相关（{lac:.3f}），可能存在未捕获的隐藏变量，建议进一步实验。"
        
        return (f"我发现了控制方程：\n  {phys_eq}\n"
                f"拟合优度 R² = {r2:.4f}。{reliability}")

    def _format_equation_physically(self, equation, target_column=None):
        """将方程表达式重写为目标变量的物理形式，并美化符号。"""
        # 1. 基础符号美化
        equation = equation.replace('sin_t', 'sin(t)').replace('cos_t', 'cos(t)')
        equation = equation.replace('exp_neg_', 'exp(-').replace('_sq', '²')
        equation = equation.replace('sqrt_', '√(')
        # 如果符号后有括号缺失，可以补上，这里先简单处理

        # 2. 处理恒等式标记
        is_identity = "(identity)" in equation
        clean_eq = equation.replace(" (identity)", "").strip()
        
        # 3. 如果提供了目标列名，重写为 d²x/dt² = ... 的形式
        if target_column and is_identity:
            lhs = self._col_to_readable(target_column)
            parts = clean_eq.split('=')
            if len(parts) == 2:
                rhs = parts[1].strip()
                return f"{lhs} = {rhs}"
            elif len(parts) == 1:
                rhs = clean_eq.replace("y = ", "").replace("y=", "")
                return f"{lhs} = {rhs}"
        
        # 默认返回美化后的方程
        return clean_eq

    def _col_to_readable(self, col):
        """将列名转换为可读的物理符号。"""
        mapping = {
            "d2x_dt2": "d²x/dt²",
            "dx_dt": "dx/dt",
            "x": "x",
            "t": "t"
        }
        return mapping.get(col, col)


# ---------- 云端智能大脑 (需 openai 包及环境变量) ----------
class GPTBrain:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        import openai
        self.openai = openai
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY")
        self.openai.api_key = self.api_key

    def parse_question(self, question, available_columns):
        prompt = (
            f"你是一个科学数据分析助手。用户有一份数据，包含以下列：{', '.join(available_columns)}。\n"
            f"用户的问题是：“{question}”\n"
            f"请从列名中选出与问题最相关的一个列名，只输出列名本身，不要输出任何其他内容。"
        )
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        answer = response.choices[0].message.content.strip()
        if answer in available_columns:
            return answer
        else:
            fallback = LocalBrain()
            return fallback.parse_question(question, available_columns)

    def generate_answer(self, result, target_column=None):
        if "error" in result:
            return f"抱歉，分析遇到问题：{result['error']}"
        equation = result['equation']
        r2 = result['r2']
        p0 = result['p0_verdict']
        lac = result.get('residual_autocorr', 0.0)
        prompt = (
            f"你是一个物理学数据科学家。CDE引擎从数据中发现了一个方程：{equation}，"
            f"R² = {r2:.4f}，残差自相关 = {lac:.4f}，P0可证伪性判决：“{p0}”。\n"
            f"请用中文向用户报告这个发现，解释方程的意义，并根据P0判决给出建议。"
        )
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()


# ---------- 工厂函数 ----------
def create_brain(mode="local"):
    if mode == "local":
        return LocalBrain()
    elif mode == "gpt":
        try:
            return GPTBrain()
        except ValueError as e:
            print(f"无法加载GPT大脑: {e}，降级为本地大脑。")
            return LocalBrain()
    else:
        raise ValueError("模式仅支持 'local' 或 'gpt'")
