import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm

# ===============================
# 🔹 Load SentencePiece Tokenizer
# ===============================
sp = spm.SentencePieceProcessor()
sp.load("empathetic_tokenizer.model")

# ===============================
# 🔹 Define Transformer (custom)
# ===============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B, Lq, D = query.shape
        Lk = key.size(1)
        q = self.W_q(query).view(B, self.num_heads, Lq, self.head_dim)
        k = self.W_k(key).view(B, self.num_heads, Lk, self.head_dim)
        v = self.W_v(value).view(B, self.num_heads, Lk, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.W_o(attn_output)



class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, dim_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, dim_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_out)
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + cross_attn_out)
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dim_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        x = self.embed(src)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dim_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(tgt)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc_out(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_layers=2, num_heads=2, dim_ff=512):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_layers, num_heads, dim_ff)
        self.decoder = Decoder(tgt_vocab, d_model, num_layers, num_heads, dim_ff)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        return self.decoder(tgt, memory, src_mask, tgt_mask)


# ===============================
# 🔹 Load trained model (CPU)
# ===============================
device = torch.device("cpu")

#model = Transformer(len(sp), len(sp), d_model=256, num_layers=2, num_heads=2, dim_ff=512).to(device)
model = Transformer(len(sp), len(sp), d_model=256, num_layers=2, num_heads=2, dim_ff=256).to(device)
#model.load_state_dict(torch.load("empathetic_transformer_final.pt", map_location=device))
model.load_state_dict(torch.load("best_transformer.pt", map_location=device))
model.eval()

# ===============================
# 🔹 Helper functions
# ===============================
def encode_and_pad(text, max_len=128):
    ids = [sp.bos_id()] + sp.encode(text, out_type=int) + [sp.eos_id()]
    if len(ids) < max_len:
        ids += [sp.pad_id()] * (max_len - len(ids))
    return torch.tensor(ids[:max_len]).unsqueeze(0).to(device)


@torch.no_grad()
def greedy_decode(model, src, sp, max_len=50):
    src_mask = torch.ones((1, 1, src.size(1)), dtype=torch.bool, device=device)
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[sp.bos_id()]], device=device)

    for _ in range(max_len - 1):
        tgt_len = ys.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device)).unsqueeze(0)
        out = model.decode(ys, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        next_token_logits = out[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_word = torch.argmax(probs, dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == sp.eos_id():
            break

    return sp.decode(ys.squeeze(0).tolist()[1:-1])

# ===============================
# 🌸 Streamlit Interface
# ===============================
st.title("🧠 Empathetic Dialogue Generator (CPU)")
st.write("Generate emotionally aware agent responses using your fine-tuned Transformer model.")
emotions = [
    "happy", "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", "ashamed",
    "caring", "confident", "content", "devastated", "disappointed", "disgusted",
    "embarrassed", "excited", "faithful", "furious", "grateful", "guilty", "hopeful",
    "impressed", "jealous", "joyful", "lonely", "nostalgic", "prepared", "proud",
    "sad", "sentimental", "surprised", "terrified", "trusting"
]

emotion = st.selectbox("Emotion", emotions)
situation = st.text_area("Situation:", "")
customer = st.text_area("Customer says:", "")

if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        input_text = f"Emotion: {emotion} | Situation: {situation} | Customer: {customer} Agent:"
        encoded = encode_and_pad(input_text)
        response = greedy_decode(model, encoded, sp)
    st.subheader("💬 Agent Response:")
    st.success(response)
# -------------------------------