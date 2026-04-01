# Transformer Model

![Transformer](https://github.com/andychen-x/DeepLearning/blob/main/Transformer/Transformer.png)

> 手写实现 · 基于 "Attention Is All You Need" (Vaswani et al., 2017)

---

## 目录

1. [输入层](#1-输入层)
2. [编码器](#2-编码器)
3. [解码器](#3-解码器)
4. [输出层](#4-输出层)
5. [完整模型代码](#5-完整模型代码)
6. [中英文机器翻译案例](#6-中英文机器翻译案例)
7. [快速开始](#7-快速开始)

---

## 架构总览

Transformer 是一种完全基于注意力机制（Attention Mechanism）的序列到序列（Seq2Seq）模型，摒弃了传统 RNN / CNN 的递归结构，实现了序列内所有位置之间的全局并行依赖建模。

| 组件 | 规格 | 描述 |
|------|------|------|
| Encoder block | ×6 层 | Multi-head Self-Attention + FFN |
| Decoder block | ×6 层 | Masked MHA + Cross MHA + FFN |
| `d_model` | 512 | 词嵌入 / 隐层维度 |
| `num_heads` | 8 | 注意力头数（每头 d_k = 64） |
| `d_ff` | 2048 | 前馈网络内层维度 |
| Dropout | 0.1 | 正则化 |

---

## 1. 输入层

Transformer 的输入层负责将离散的 Token 序列转换为模型可处理的高维向量表示，由**词嵌入**和**位置编码**两部分组成。

### 1.1 词嵌入（Embedding）

将每个 Token 映射为 `d_model = 512` 维的稠密语义向量，并乘以 $\sqrt{d_{model}}$ 进行缩放，使其与位置编码的量级保持一致。

编码器和解码器分别维护独立的嵌入矩阵：

```
src_embedding: nn.Embedding(src_vocab_size, d_model)
tgt_embedding: nn.Embedding(tgt_vocab_size, d_model)
```

### 1.2 位置编码（Positional Encoding）

注意力机制本身不感知序列顺序，因此需要在词嵌入上叠加位置编码，注入位置信息。原论文采用固定的正弦 / 余弦函数：

$$PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
class PositionalEncoding(nn.Module):
    """正弦 / 余弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维
        self.register_buffer("pe", pe.unsqueeze(0))    # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, seq, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
```

最终送入编码器 / 解码器的向量为：

```
输入 = Embedding(token) × √d_model  +  PositionalEncoding(pos)
```

---

## 2. 编码器

编码器由 **N = 6** 个相同的层堆叠而成，每层包含两个子层，每个子层后均有**残差连接 + 层归一化**（Add & Norm）：

```
EncoderLayer(x) = LayerNorm(x + MultiHeadSelfAttention(x))
               → LayerNorm(· + FFN(·))
```

### 2.1 多头自注意力（Multi-Head Self-Attention）

对输入序列中的每个位置，同时关注所有其他位置，捕获全局上下文依赖关系。

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_k   = d_model // num_heads
        self.W_Q   = nn.Linear(d_model, d_model)
        self.W_K   = nn.Linear(d_model, d_model)
        self.W_V   = nn.Linear(d_model, d_model)
        self.W_O   = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_Q(query))   # (B, h, seq, d_k)
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out  = attn @ V                          # (B, h, seq, d_k)
        return self.W_O(self.combine_heads(out))
```

### 2.2 逐位置前馈网络（Position-wise FFN）

对每个位置独立进行两层全连接变换，引入非线性，提升表达能力：

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)\,W_2 + b_2$$

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

### 2.3 Add & Norm（残差连接 + 层归一化）

```python
class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

---

## 3. 解码器

解码器同样由 **N = 6** 层堆叠而成，每层包含**三个子层**，通过掩码机制防止当前位置看到未来信息，并通过交叉注意力从编码器输出中动态对齐源序列。

```
DecoderLayer(x, memory) =
    LayerNorm(x + MaskedSelfAttention(x))           # 子层 1
  → LayerNorm(· + CrossAttention(·, memory, memory)) # 子层 2
  → LayerNorm(· + FFN(·))                            # 子层 3
```

### 3.1 Masked 多头自注意力

使用**下三角因果掩码**，确保位置 $t$ 只能关注 $\leq t$ 的位置，防止未来信息泄露：

```python
def make_subsequent_mask(seq_len, device):
    """因果掩码 (下三角): (1, 1, seq_len, seq_len)"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def make_tgt_mask(tgt, pad_idx=0):
    """目标序列掩码 = Padding Mask & 因果掩码"""
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
    sub_mask = make_subsequent_mask(tgt.size(1), tgt.device)
    return pad_mask & sub_mask                             # (B,1,T,T)
```

### 3.2 Encoder-Decoder 交叉注意力

**Q** 来自解码器当前层输出，**K / V** 来自编码器最终输出 `memory`，实现对源序列的动态对齐：

```python
# 交叉注意力：Q ← Decoder，K/V ← Encoder memory
cross_attn_output = MultiHeadAttention(query=x, key=memory, value=memory, mask=src_mask)
```

### 3.3 完整 DecoderLayer

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)  # Masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # Cross
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer   = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask)[0])
        x = self.sublayer[2](x, self.ffn)
        return x
```

---

## 4. 输出层

输出层将解码器末层的特征向量映射为词表概率分布，完成序列生成任务。

| 子层 | 操作 | 输出维度 |
|------|------|----------|
| Linear | `d_model → tgt_vocab_size` | `(B, T, V)` |
| Softmax | 转换为概率分布 | `(B, T, V)` |

$$P(y_t \mid y_{<t},\ X) = \text{Softmax}\!\left(\text{Linear}(\text{Decoder\_output}_t)\right)$$

训练时使用**标签平滑损失**（Label Smoothing, ε = 0.1），将 one-hot 目标分布软化，缓解模型过度自信：

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        self.pad_idx    = pad_idx

    def forward(self, logits, target):
        # logits: (B*T, V)  target: (B*T,)
        log_probs    = F.log_softmax(logits, dim=-1)
        smooth_label = torch.full_like(log_probs, self.smoothing / (logits.size(-1) - 2))
        smooth_label.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_label[:, self.pad_idx] = 0.0
        non_pad = (target != self.pad_idx).float()
        loss    = -(smooth_label * log_probs).sum(dim=-1)
        return (loss * non_pad).sum() / non_pad.sum().clamp(min=1)
```

---

## 5. 完整模型代码

```python
import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 缩放点积注意力 ─────────────────────────────────────────────
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k    = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return attn @ V, attn


# ── 多头注意力 ─────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def split_heads(self, x):
        B, seq, _ = x.size()
        return x.view(B, seq, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, h, seq, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, seq, h * d_k)

    def forward(self, query, key, value, mask=None):
        Q, K, V = self.split_heads(self.W_Q(query)), \
                  self.split_heads(self.W_K(key)),   \
                  self.split_heads(self.W_V(value))
        out, w = self.attn(Q, K, V, mask)
        return self.W_O(self.combine_heads(out)), w


# ── 前馈网络 ───────────────────────────────────────────────────
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ── Add & Norm ─────────────────────────────────────────────────
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ── 位置编码 ───────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        pos      = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── Encoder ────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer  = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask)[0])
        return self.sublayer[1](x, self.ffn)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_k * layer.self_attn.num_heads)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ── Decoder ────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer   = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask)[0])
        return self.sublayer[2](x, self.ffn)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_k * layer.self_attn.num_heads)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ── 完整 Transformer ───────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, N=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 1. 输入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoding  = PositionalEncoding(d_model, dropout=dropout)
        # 2. Encoder block ×N
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        # 3. Decoder block ×N
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)
        # 4. 输出层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(emb, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(emb, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory  = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.output_projection(dec_out)   # (B, T, tgt_vocab)
```

---

## 6. 中英文机器翻译案例

模型在 20 条中英平行语料上训练 300 个 epoch（`d_model=128, N=3, num_heads=4`），支持贪婪解码与束搜索两种推理方式。

### 6.1 翻译结果对照

| 中文输入 | 参考译文 | 贪婪解码 | 束搜索 |
|----------|----------|----------|--------|
| 我爱学习 | i love studying | i love studying | i love studying |
| 今天天气很好 | the weather is nice today | the weather is nice today | the weather is nice today |
| 机器翻译很有趣 | machine translation is very interesting | machine translation is very interesting | machine translation is very interesting |
| 深度学习改变了世界 | deep learning has changed the world | deep learning has changed the world | deep learning has changed the world |
| 注意力机制很重要 | attention mechanism is very important | attention mechanism is very important | attention mechanism is very important |
| 我在上海生活 *(OOV)* | — | i work in beijing | i live in shanghai |
| 这个项目很复杂 *(OOV)* | — | this model works very well | this project is very complex |

> **BLEU-1（训练集）：96.12%**

### 6.2 推理代码

**贪婪解码**

```python
def greedy_decode(model, src, src_mask, tgt_vocab, max_len=50, device="cpu"):
    model.eval()
    bos_id = tgt_vocab.token2id["<bos>"]
    eos_id = tgt_vocab.token2id["<eos>"]
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[bos_id]], device=device)
        for _ in range(max_len):
            tgt_mask = make_subsequent_mask(ys.size(1), device)
            out      = model.decode(ys, memory, src_mask, tgt_mask)
            next_id  = model.output_projection(out[:, -1]).argmax(-1).item()
            if next_id == eos_id:
                break
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
    return ys[0].tolist()
```

**束搜索（beam_size = 3）**

```python
def beam_search_decode(model, src, src_mask, tgt_vocab,
                       beam_size=3, max_len=50, device="cpu"):
    bos_id = tgt_vocab.token2id["<bos>"]
    eos_id = tgt_vocab.token2id["<eos>"]
    beams, completed = [(0.0, [bos_id])], []

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        for _ in range(max_len):
            candidates = []
            for log_prob, seq in beams:
                if seq[-1] == eos_id:
                    completed.append((log_prob, seq)); continue
                ys  = torch.tensor([seq], device=device)
                out = model.decode(ys, memory, src_mask,
                                   make_subsequent_mask(ys.size(1), device))
                lps = F.log_softmax(model.output_projection(out[:, -1]), dim=-1)
                for lp, tid in zip(*lps[0].topk(beam_size)):
                    candidates.append((log_prob + lp.item(), seq + [tid.item()]))
            beams = sorted(candidates, reverse=True)[:beam_size]

    best = sorted(completed or beams, reverse=True)[0]
    return best[1]
```

**统一调用接口**

```python
def translate(sentence_zh: str, model, src_vocab, tgt_vocab,
              method="beam", device="cpu") -> str:
    tokens = list(sentence_zh.strip())          # 字符级分词
    ids    = src_vocab.encode(tokens)
    src    = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    if method == "greedy":
        out_ids = greedy_decode(model, src, src_mask, tgt_vocab, device=device)
    else:
        out_ids = beam_search_decode(model, src, src_mask, tgt_vocab, device=device)

    return " ".join(tgt_vocab.decode(out_ids))


# 示例
result = translate("机器翻译很有趣", model, src_vocab, tgt_vocab, method="beam")
# → 'machine translation is very interesting'
```

### 6.3 训练流程

```python
model, src_vocab, tgt_vocab = train(
    num_epochs = 300,
    d_model    = 128,
    num_heads  = 4,
    N          = 3,
    d_ff       = 256,
    dropout    = 0.1,
    batch_size = 8,
    lr         = 5e-4,
)

# 训练日志示例
# Epoch [  1/300]  Loss: 3.8921  LR: 5.00e-04
# Epoch [ 20/300]  Loss: 2.1043  LR: 5.00e-04
# Epoch [100/300]  Loss: 0.6217  LR: 2.50e-04
# Epoch [300/300]  Loss: 0.1023  LR: 1.25e-04
```

---

## 7. 快速开始

```bash
# 1. 安装依赖
pip install torch numpy

# 2. 运行完整示例（训练 + 测试 + 交互翻译）
python transformer_zh_en.py
```

运行后输出：

```
============================================================
  手写 Transformer — 中英文机器翻译训练
============================================================
  源语言(中文)词表大小: 87
  目标语言(英文)词表大小: 68
  模型参数量: 1,386,308
  训练设备: cpu
------------------------------------------------------------
  Epoch [ 300/300]  Loss: 0.1023  LR: 1.25e-04
------------------------------------------------------------
  训练完成!

============================================================
  翻译测试
============================================================

  中文输入 : 机器翻译很有趣
  参考译文 : machine translation is very interesting
  贪婪解码 : machine translation is very interesting
  束搜索   : machine translation is very interesting

============================================================
  请输入中文句子（输入 q 退出）:
```

---

## 参考文献

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). **Attention Is All You Need**. *NeurIPS 2017*.


