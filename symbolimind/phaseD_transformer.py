"""
Phase D: Grammar-Constrained Transformer
在 Physics Grammar 约束下生成合法物理结构的轻量级 Transformer。
"""
import torch
import torch.nn as nn
import math, random, numpy as np

# 词表
ATOMS = ['t1','o1','t2','o2','d','sin_t1','cos_t1','sin_t2','cos_t2','sin_d','cos_d','o1_sq','o2_sq']
FUNCS = ['sin','cos','sq']
OPS = ['+','-','*']
SPECIALS = ['(', ')', '<S>', '<E>', '<PAD>']
VOCAB = SPECIALS + OPS + FUNCS + ATOMS
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

class PhysicsTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask)
        return self.output(x)

class GrammarConstraint:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
    
    def is_valid(self, tokens):
        depth = 0
        max_d = 0
        for t in tokens:
            if t == '(': depth += 1; max_d = max(max_d, depth)
            elif t == ')': depth -= 1
        return depth == 0 and max_d <= self.max_depth

def tokenize(expr):
    tokens = ['<S>']
    for word in expr.replace('*',' * ').replace('+',' + ').replace('-',' - ').split():
        tokens.append(word)
    tokens.append('<E>')
    return [TOKEN_TO_ID[t] for t in tokens]

def generate_structure(model, max_len=15, temperature=0.8):
    grammar = GrammarConstraint(max_depth=3)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        seq = [TOKEN_TO_ID['<S>']]
        for _ in range(max_len):
            seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)
            output = model(seq_tensor)
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            for i in range(VOCAB_SIZE):
                tok = ID_TO_TOKEN[i]
                test_tokens = [ID_TO_TOKEN[t] for t in seq[1:]] + [tok]
                if tok == '<E>' and not grammar.is_valid([ID_TO_TOKEN[t] for t in seq[1:]]):
                    probs[i] = 0
                elif tok == '<PAD>':
                    probs[i] = 0
            if probs.sum() == 0: break
            next_token = torch.multinomial(probs, 1).item()
            if next_token == TOKEN_TO_ID['<E>']: break
            seq.append(next_token)
        tokens = [ID_TO_TOKEN[t] for t in seq if t not in [TOKEN_TO_ID['<S>'], TOKEN_TO_ID['<E>'], TOKEN_TO_ID['<PAD>']]]
        return ' '.join(tokens)
