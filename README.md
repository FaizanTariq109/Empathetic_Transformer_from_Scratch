# Empathetic Transformer for Dialogue Generation

This project implements a Transformer-based encoder–decoder model **from scratch in PyTorch** for empathetic dialogue generation.  
The model is trained on the **EmpatheticDialogues** dataset and is capable of generating emotionally aware and contextually relevant agent responses given an emotion, situation, and user utterance.

---

## 📌 Project Overview

- Custom Transformer (no `nn.Transformer`)
- Multi-head self-attention and cross-attention implemented manually
- SentencePiece BPE tokenizer
- Emotion-conditioned dialogue generation
- Trained and evaluated on the EmpatheticDialogues dataset
- Deployed locally using Streamlit (CPU inference)

---

## 🗂 Dataset

**EmpatheticDialogues** (Rashkin et al., ACL 2019)

- Total samples: ~64,000
- Unique emotions: 32
- Splits:
  - 80% Train
  - 10% Validation
  - 10% Test

Each sample contains:

- Emotion label
- Situation description
- User utterance
- Empathetic agent response

---

## 🔤 Tokenization

- Tokenizer: **SentencePiece (BPE)**
- Vocabulary size: 8000
- Special tokens:
  - `<PAD>`
  - `<BOS>`
  - `<EOS>`
  - `<UNK>`
- Maximum sequence length: 128

SentencePiece was used for subword-level tokenization and robust handling of unseen words.

---

## 🧠 Model Architecture

Transformer Encoder–Decoder implemented from scratch.

### Architecture Configuration

| Parameter                  | Value |
| -------------------------- | ----- |
| Embedding size (`d_model`) | 256   |
| Attention heads            | 2     |
| Encoder layers             | 2     |
| Decoder layers             | 2     |
| Feed-forward dimension     | 512   |
| Dropout                    | 0.1   |
| Max sequence length        | 128   |

---

## ⚙️ Training Setup

- Optimizer: Adam
- Learning rate: 1e-4
- Loss function: CrossEntropy with label smoothing (0.1)
- Batch size: 32
- Epochs: 5
- Gradient clipping: 1.0
- Learning rate scheduler: ReduceLROnPlateau
- Mixed-precision training (AMP)
- Best checkpoint saved based on validation BLEU

Training was performed on **Tesla T4 GPU (Kaggle)**.

---

## 📊 Evaluation Metrics

The model was evaluated using the following metrics:

- **BLEU** – n-gram precision
- **ROUGE-L** – longest common subsequence
- **chrF** – character-level F1 score
- **Perplexity (PPL)** – fluency measure

---

## 📈 Final Results

| Metric     | Score  |
| ---------- | ------ |
| BLEU       | 0.0116 |
| ROUGE-L    | 0.1329 |
| chrF       | 12.11  |
| Perplexity | 63.76  |

---

## 🧪 Qualitative Example

**Emotion:** hopeful  
**Situation:** I have been preparing for months for this job interview tomorrow.  
**User:** I am nervous but excited for what is ahead.

**Generated Response:**

> It’s completely natural to feel that way, and all your preparation will help you do your best.

---

## 🚀 Inference & Deployment

- Inference uses **greedy decoding**
- No teacher forcing during inference
- Masking applied to prevent future token leakage
- Deployed locally using **Streamlit**
- Runs on CPU (no GPU required)

---

## 🧩 Key Concepts Used

- Multi-head self-attention
- Cross-attention in decoder
- Positional encoding
- Teacher forcing (training only)
- Causal masking
- Label smoothing
- Greedy decoding

---

## 🧠 Known Limitations

- Low BLEU due to open-ended nature of dialogue generation
- Occasional generic or repetitive responses
- No pre-trained embeddings used
- Limited training epochs

---

## 🔮 Future Improvements

- Use pre-trained language models (BERT / GPT)
- Increase model depth and heads
- Emotion-specific decoding control
- Beam search decoding
- Longer training schedule

---

## 👤 Author

**Faizan Tariq**  
FAST-NUCES  
Bachelor of Software Engineering
