import os
import json
import torch
import faiss
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model
from tokenizers import ByteLevelBPETokenizer
from flash_attn import flash_attn_func

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- CONFIG ----------------
block_size = 128
n_embd = 1024
n_head = 8
n_layer = 12
dropout = 0.1

# ---------------- MODEL COMPONENTS ----------------
class FlashMHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).half()
        return flash_attn_func(qkv[0], qkv[1], qkv[2], causal=True).contiguous().view(B, T, C)

class RAGBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = FlashMHA(dim, heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ctx_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x, ctx_emb):
        x = x + self.attn(self.norm1(x))
        gate = torch.sigmoid(self.ctx_gate(ctx_emb)) * 2
        x = x + gate * ctx_emb
        x = x + self.ffn(self.norm2(x))
        return x

class FAISSRetriever(nn.Module):
    def __init__(self, index_path="faiss_index"):
        super().__init__()
        self.device = device

        # Load saved docs and index
        with open("faiss_index/docs.json", "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))

        # Load LoRA-augmented embedding model
        word_embedding_model = Transformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            model_args={'add_pooling_layer': False}
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["attention.self.query", "attention.self.key", "attention.self.value"],
            lora_dropout=0.05,
            bias="none"
        )

        word_embedding_model.auto_model = get_peft_model(word_embedding_model.auto_model, lora_config)

        self.embed_model = SentenceTransformer(modules=[
            word_embedding_model,
            Pooling(word_embedding_model.get_word_embedding_dimension())
        ]).to(self.device)

    def retrieve(self, queries, k=3):
        query_embs = self.embed_model.encode(
            queries,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        _, indices = self.index.search(query_embs.astype(np.float32), k)
        return [[self.docs[i] for i in batch] for batch in indices]

class RAGGPT(nn.Module):
    def __init__(self, vocab_size, retriever):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.ctx_proj = nn.Linear(retriever.embed_model.get_sentence_embedding_dimension(), n_embd)
        self.retriever = retriever
        self.blocks = nn.Sequential(*[RAGBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        queries = [decode(seq.tolist()) for seq in idx]
        ctx = self.retriever.retrieve(queries)
        ctx_emb = torch.stack([
            self.retriever.embed_model.encode(c, convert_to_tensor=True).to(device)
            for c in ctx
        ])
        ctx_emb = self.ctx_proj(ctx_emb).mean(dim=1).unsqueeze(1).expand(-1, T, -1)
        
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = tok_emb + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x, ctx_emb)
        return self.head(self.ln(x))

    def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
        self.eval()
        with torch.no_grad():
            idx = torch.tensor(encode(prompt), device=device).unsqueeze(0)
            for _ in range(max_new_tokens):
                logits = self(idx[:, -block_size:])
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return decode(idx[0].tolist())

# ---------------- TOKENIZER + LOAD ----------------
def load_inference_model(checkpoint_path="checkpoint.pth"):
    tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    global encode, decode
    encode = lambda s: tokenizer.encode(s).ids
    decode = lambda l: tokenizer.decode(l)

    retriever = FAISSRetriever("faiss_index")
    model = RAGGPT(tokenizer.get_vocab_size(), retriever).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

# ---------------- RUN INFERENCE ----------------
if __name__ == "__main__":
    model = load_inference_model()
    prompt = "भारत"
    print("Generated:\n")
    print(model.generate(prompt, max_new_tokens=200))
