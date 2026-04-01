"""
手写 Transformer 模型 — 中英文机器翻译
架构参考: "Attention Is All You Need" (Vaswani et al., 2017)
对应图示: Transformer 模型架构 (N=6)

模块结构:
  1. 输入  — Input/Output Embedding + Positional Encoding
  2. Encoder block — Multi-head Attention + Feed Forward (×N)
  3. Decoder block — Masked MHA + Cross MHA + Feed Forward (×N)
  4. 输出  — Linear + Softmax
"""

import math
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 基础组件
# ============================================================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力 Scaled Dot-Product Attention"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_v, d_v)
        mask: (batch, 1, seq_q, seq_k)  — 可选
        """
        d_k = Q.size(-1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, seq_q, seq_k)

        # 掩码 — 将 mask=0 的位置填充为极小值
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)   # (B, h, seq_q, seq_k)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)     # (B, h, seq_q, d_v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力 Multi-Head Attention"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 每个头的维度

        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def split_heads(self, x):
        """(B, seq, d_model) → (B, h, seq, d_k)"""
        B, seq, _ = x.size()
        x = x.view(B, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """(B, h, seq, d_k) → (B, seq, d_model)"""
        B, h, seq, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(B, seq, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))

        attn_output, attn_weights = self.attention(Q, K, V, mask)

        output = self.combine_heads(attn_output)
        output = self.W_O(output)
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络 Position-wise Feed-Forward Network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """层归一化 Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    """残差连接 + LayerNorm (Add & Norm)"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm    = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Pre-Norm 变体; 原论文为 Post-Norm"""
        return x + self.dropout(sublayer(self.norm(x)))


# ============================================================
# 2. 位置编码 Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 构建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (B, seq, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ============================================================
# 3. Encoder 编码器
# ============================================================

class EncoderLayer(nn.Module):
    """单层 Encoder = Self-Attention + FFN + 两个 Add&Norm"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer  = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask)[0])
        x = self.sublayer[1](x, self.ffn)
        return x


class Encoder(nn.Module):
    """N 层 Encoder"""

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = LayerNorm(layer.self_attn.d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ============================================================
# 4. Decoder 解码器
# ============================================================

class DecoderLayer(nn.Module):
    """单层 Decoder = Masked Self-Attention + Cross-Attention + FFN + 三个 Add&Norm"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)  # Masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # Encoder-Decoder
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer   = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 1. Masked Multi-head Self-Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        # 2. Cross Multi-head Attention (Q←Decoder, K/V←Encoder memory)
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask)[0])
        # 3. Feed Forward
        x = self.sublayer[2](x, self.ffn)
        return x


class Decoder(nn.Module):
    """N 层 Decoder"""

    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = LayerNorm(layer.self_attn.d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ============================================================
# 5. 完整 Transformer 模型
# ============================================================

class Transformer(nn.Module):
    """
    完整 Transformer 模型
    对应架构图中的 1~4 四个区域
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        N: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. 输入 — Embedding + Positional Encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoding  = PositionalEncoding(d_model, max_len, dropout)

        # 2. Encoder block (×N)
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder  = Encoder(encoder_layer, N)

        # 3. Decoder block (×N)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder  = Decoder(decoder_layer, N)

        # 4. 输出 — Linear + Softmax (在 loss 中已含 Softmax)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(tgt_emb, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)
        logits = self.output_projection(dec_out)  # (B, seq, tgt_vocab)
        return logits


# ============================================================
# 6. 掩码生成工具
# ============================================================

def make_pad_mask(seq, pad_idx=0):
    """生成 Padding 掩码: (B, 1, 1, seq_len)"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def make_subsequent_mask(seq_len, device):
    """生成因果掩码 (下三角): (1, 1, seq_len, seq_len)"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def make_tgt_mask(tgt, pad_idx=0):
    """目标序列掩码 = Padding 掩码 & 因果掩码"""
    pad_mask  = make_pad_mask(tgt, pad_idx)                          # (B,1,1,T)
    sub_mask  = make_subsequent_mask(tgt.size(1), tgt.device)        # (1,1,T,T)
    return pad_mask & sub_mask                                        # (B,1,T,T)


# ============================================================
# 7. 简易词表 & 分词器
# ============================================================

class SimpleVocab:
    """字符级 / 词级 简易词表"""

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self):
        self.token2id = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}
        self.id2token = {v: k for k, v in self.token2id.items()}

    def build(self, sentences):
        for sent in sentences:
            for tok in sent:
                if tok not in self.token2id:
                    idx = len(self.token2id)
                    self.token2id[tok] = idx
                    self.id2token[idx] = tok
        return self

    def encode(self, tokens, add_bos=True, add_eos=True):
        ids = [self.token2id.get(t, self.token2id[self.UNK]) for t in tokens]
        if add_bos: ids = [self.token2id[self.BOS]] + ids
        if add_eos: ids = ids + [self.token2id[self.EOS]]
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, self.UNK)
            if tok in (self.PAD, self.BOS): continue
            if tok == self.EOS: break
            tokens.append(tok)
        return tokens

    def __len__(self):
        return len(self.token2id)


def zh_tokenize(sent: str):
    """简易中文分词: 字符级"""
    return list(sent.strip())

def en_tokenize(sent: str):
    """简易英文分词: 空格分词 + 小写"""
    return sent.strip().lower().split()


# ============================================================
# 8. 数据集
# ============================================================

# 示例平行语料 (中 → 英)
PARALLEL_DATA = [
    ("我爱学习", "i love studying"),
    ("你好世界", "hello world"),
    ("今天天气很好", "the weather is nice today"),
    ("机器翻译很有趣", "machine translation is very interesting"),
    ("深度学习改变了世界", "deep learning has changed the world"),
    ("这是一个测试", "this is a test"),
    ("我喜欢吃苹果", "i like eating apples"),
    ("明天我去学校", "i will go to school tomorrow"),
    ("人工智能发展很快", "artificial intelligence is developing rapidly"),
    ("他正在读一本书", "he is reading a book"),
    ("我们需要更多数据", "we need more data"),
    ("神经网络很强大", "neural networks are very powerful"),
    ("请问你叫什么名字", "what is your name"),
    ("中文很难学", "chinese is difficult to learn"),
    ("英语是国际语言", "english is an international language"),
    ("我在北京工作", "i work in beijing"),
    ("今天是星期一", "today is monday"),
    ("这个模型效果很好", "this model works very well"),
    ("注意力机制很重要", "attention mechanism is very important"),
    ("编码器解码器架构", "encoder decoder architecture"),
]


class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tok_fn, tgt_tok_fn, max_len=50):
        self.samples = []
        for src_sent, tgt_sent in data:
            src_ids = src_vocab.encode(src_tok_fn(src_sent))
            tgt_ids = tgt_vocab.encode(tgt_tok_fn(tgt_sent))
            if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                self.samples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_id=0):
    """填充到同批次最大长度"""
    src_batch, tgt_batch = zip(*batch)

    max_src = max(len(s) for s in src_batch)
    max_tgt = max(len(t) for t in tgt_batch)

    src_padded = [s + [pad_id] * (max_src - len(s)) for s in src_batch]
    tgt_padded = [t + [pad_id] * (max_tgt - len(t)) for t in tgt_batch]

    return (
        torch.tensor(src_padded, dtype=torch.long),
        torch.tensor(tgt_padded, dtype=torch.long),
    )


# ============================================================
# 9. 标签平滑损失
# ============================================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        logits : (B*T, V)
        target : (B*T,)
        """
        B_T, V = logits.size()
        log_probs = F.log_softmax(logits, dim=-1)

        # 构建平滑标签
        smooth_label = torch.full((B_T, V), self.smoothing / (V - 2), device=logits.device)
        smooth_label.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_label[:, self.pad_idx] = 0.0

        # 忽略 padding
        non_pad_mask = (target != self.pad_idx).float()
        loss = -(smooth_label * log_probs).sum(dim=-1)
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum().clamp(min=1)
        return loss


# ============================================================
# 10. 训练
# ============================================================

def build_vocabs():
    """从平行语料构建词表"""
    src_sents = [zh_tokenize(zh) for zh, _ in PARALLEL_DATA]
    tgt_sents = [en_tokenize(en) for _, en in PARALLEL_DATA]

    src_vocab = SimpleVocab().build(src_sents)
    tgt_vocab = SimpleVocab().build(tgt_sents)
    return src_vocab, tgt_vocab


def train(
    num_epochs: int = 200,
    d_model: int = 128,
    num_heads: int = 4,
    N: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
):
    print("=" * 60)
    print("  手写 Transformer — 中英文机器翻译训练")
    print("=" * 60)

    # 构建词表
    src_vocab, tgt_vocab = build_vocabs()
    print(f"  源语言(中文)词表大小: {len(src_vocab)}")
    print(f"  目标语言(英文)词表大小: {len(tgt_vocab)}")

    # 数据集
    dataset = TranslationDataset(
        PARALLEL_DATA, src_vocab, tgt_vocab, zh_tokenize, en_tokenize
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )

    # 模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        N=N,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"  训练设备: {device}")
    print("-" * 60)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=False
    )
    criterion = LabelSmoothingLoss(len(tgt_vocab), pad_idx=0, smoothing=0.1)

    # 训练循环
    model.train()
    pad_id = src_vocab.token2id[SimpleVocab.PAD]

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Decoder 输入: [BOS, w1, ..., wn]   目标: [w1, ..., wn, EOS]
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_pad_mask(src, pad_id)
            tgt_mask = make_tgt_mask(tgt_in, pad_id)

            logits = model(src, tgt_in, src_mask, tgt_mask)

            B, T, V = logits.size()
            loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:>4}/{num_epochs}]  Loss: {avg_loss:.4f}  "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    print("-" * 60)
    print("  训练完成!")
    return model, src_vocab, tgt_vocab


# ============================================================
# 11. 推理 — 贪婪解码 & 束搜索
# ============================================================

def greedy_decode(model, src, src_mask, tgt_vocab, max_len=50, device="cpu"):
    """贪婪解码"""
    model.eval()
    bos_id = tgt_vocab.token2id[SimpleVocab.BOS]
    eos_id = tgt_vocab.token2id[SimpleVocab.EOS]

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        # 初始化解码序列
        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = make_subsequent_mask(ys.size(1), device)
            out = model.decode(ys, memory, src_mask, tgt_mask)
            logits = model.output_projection(out[:, -1, :])  # 只取最后一步
            next_id = logits.argmax(dim=-1).item()

            if next_id == eos_id:
                break
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)

    return ys[0].tolist()


def beam_search_decode(model, src, src_mask, tgt_vocab, beam_size=3, max_len=50, device="cpu"):
    """束搜索解码"""
    model.eval()
    bos_id = tgt_vocab.token2id[SimpleVocab.BOS]
    eos_id = tgt_vocab.token2id[SimpleVocab.EOS]

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        # beam: list of (log_prob, token_ids)
        beams = [(0.0, [bos_id])]
        completed = []

        for _ in range(max_len):
            candidates = []
            for log_prob, seq in beams:
                if seq[-1] == eos_id:
                    completed.append((log_prob, seq))
                    continue

                ys = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = make_subsequent_mask(ys.size(1), device)
                out = model.decode(ys, memory, src_mask, tgt_mask)
                log_probs = F.log_softmax(model.output_projection(out[:, -1, :]), dim=-1)
                top_lp, top_ids = log_probs[0].topk(beam_size)

                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append((log_prob + lp, seq + [tid]))

            # 保留得分最高的 beam_size 个
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

            if len(completed) >= beam_size:
                break

        if not completed:
            completed = beams

        best = sorted(completed, key=lambda x: x[0], reverse=True)[0]
        return best[1]


def translate(sentence_zh: str, model, src_vocab, tgt_vocab,
              method="beam", device="cpu"):
    """
    翻译单句中文 → 英文
    method: "greedy" 或 "beam"
    """
    model.eval()
    tokens = zh_tokenize(sentence_zh)
    ids = src_vocab.encode(tokens)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = make_pad_mask(src)

    if method == "greedy":
        out_ids = greedy_decode(model, src, src_mask, tgt_vocab, device=device)
    else:
        out_ids = beam_search_decode(model, src, src_mask, tgt_vocab, device=device)

    out_tokens = tgt_vocab.decode(out_ids)
    return " ".join(out_tokens)


# ============================================================
# 12. 测试 & 评估
# ============================================================

def run_tests(model, src_vocab, tgt_vocab, device="cpu"):
    print("\n" + "=" * 60)
    print("  翻译测试 (训练集内句子)")
    print("=" * 60)

    test_cases = [
        ("我爱学习", "i love studying"),
        ("今天天气很好", "the weather is nice today"),
        ("机器翻译很有趣", "machine translation is very interesting"),
        ("深度学习改变了世界", "deep learning has changed the world"),
        ("注意力机制很重要", "attention mechanism is very important"),
    ]

    for zh, ref_en in test_cases:
        greedy_out = translate(zh, model, src_vocab, tgt_vocab, method="greedy", device=device)
        beam_out   = translate(zh, model, src_vocab, tgt_vocab, method="beam",   device=device)
        print(f"\n  中文输入 : {zh}")
        print(f"  参考译文 : {ref_en}")
        print(f"  贪婪解码 : {greedy_out}")
        print(f"  束搜索   : {beam_out}")

    print("\n" + "=" * 60)
    print("  OOV / 泛化测试 (训练集外句子)")
    print("=" * 60)

    unk_cases = [
        "我在上海生活",
        "这个项目很复杂",
        "他喜欢音乐",
    ]
    for zh in unk_cases:
        out = translate(zh, model, src_vocab, tgt_vocab, method="beam", device=device)
        print(f"\n  中文输入 : {zh}")
        print(f"  束搜索   : {out}")


def simple_bleu(reference: str, hypothesis: str) -> float:
    """简易 BLEU-1 实现"""
    ref_tokens  = reference.split()
    hyp_tokens  = hypothesis.split()
    if not hyp_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    matches = sum(1 for t in hyp_tokens if t in ref_set)
    precision = matches / len(hyp_tokens)
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return bp * precision


def evaluate_bleu(model, src_vocab, tgt_vocab, data, device="cpu"):
    print("\n" + "=" * 60)
    print("  BLEU-1 评估 (训练集)")
    print("=" * 60)
    scores = []
    for zh, ref_en in data:
        hyp = translate(zh, model, src_vocab, tgt_vocab, method="beam", device=device)
        score = simple_bleu(ref_en, hyp)
        scores.append(score)
    avg = sum(scores) / len(scores)
    print(f"  平均 BLEU-1: {avg:.4f}  ({avg*100:.2f}%)")
    return avg


# ============================================================
# 13. 主程序
# ============================================================

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 超参数 ----
    CONFIG = dict(
        num_epochs = 300,
        d_model    = 128,
        num_heads  = 4,
        N          = 3,       # 论文用 N=6; 这里演示用 N=3
        d_ff       = 256,
        dropout    = 0.1,
        batch_size = 8,
        lr         = 5e-4,
        device     = DEVICE,
    )

    # ---- 训练 ----
    t0 = time.time()
    model, src_vocab, tgt_vocab = train(**CONFIG)
    print(f"\n  总训练时长: {time.time()-t0:.1f}s")

    # ---- 翻译测试 ----
    run_tests(model, src_vocab, tgt_vocab, device=DEVICE)

    # ---- BLEU 评估 ----
    evaluate_bleu(model, src_vocab, tgt_vocab, PARALLEL_DATA, device=DEVICE)

    # ---- 交互式翻译 ----
    print("\n" + "=" * 60)
    print("  交互式翻译 (输入 'q' 退出)")
    print("=" * 60)
    while True:
        try:
            user_input = input("\n  请输入中文句子: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("q", "quit", "exit", ""):
            break
        result = translate(user_input, model, src_vocab, tgt_vocab,
                           method="beam", device=DEVICE)
        print(f"  英文翻译    : {result}")

    print("\n  程序结束。")
