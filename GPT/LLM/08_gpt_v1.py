# ---------------------- OPTIMIZED 400M PARAM RAGGPT ----------------------
import torch
import torch.nn as nn
import json, os, faiss
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from sentence_transformers import SentenceTransformer
from flash_attn import flash_attn_func
from lion_pytorch import Lion

# Config
device = 'cuda'
batch_size = 96
block_size = 192
max_iters = 3500
learning_rate = 2e-4
n_embd = 1024
n_head = 16
n_layer = 24
dropout = 0.05
warmup_iters = 500

# ---------------------- OPTIMIZED DATA LOADING ----------------------
class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        
    def __len__(self): 
        return len(self.data) - block_size
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+block_size+1].astype(np.int64))
        return x, y

# ---------------------- GPU-ACCELERATED RETRIEVER ----------------------
class FAISSRetriever:
    def __init__(self, docs, embed_model):
        self.docs = docs
        self.embed_model = embed_model
        self.res = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatL2(
            self.res,
            embed_model.get_sentence_embedding_dimension()
        )
        self._build_index()
        
    def _build_index(self):
        embs = self.embed_model.encode(self.docs, batch_size=512, show_progress_bar=True)
        self.index.add(np.array(embs).astype('float32'))
        
    def retrieve(self, queries, k=3):
        query_embs = self.embed_model.encode(queries)
        _, indices = self.index.search(query_embs, k)
        return [[self.docs[i] for i in batch] for batch in indices]

# ---------------------- OPTIMIZED MODEL ARCH ----------------------
class FlashMHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x).view(B,T,3,self.heads,C//self.heads).permute(2,0,3,1,4)
        return flash_attn_func(qkv[0], qkv[1], qkv[2], causal=True).contiguous().view(B,T,C)

class RAGBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = FlashMHA(dim, heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + torch.utils.checkpoint.checkpoint(self.attn, self.norm1(x))
        x = x + torch.utils.checkpoint.checkpoint(self.ffn, self.norm2(x))
        return x

class RAGGPT(nn.Module):
    def __init__(self, vocab_size, retriever):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size+1, n_embd)
        self.ctx_proj = nn.Linear(768, n_embd)
        self.retriever = retriever
        self.blocks = nn.Sequential(*[RAGBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        # Retrieve & project context
        with torch.no_grad():
            ctx = self.retriever.retrieve([decode(seq) for seq in idx])
            ctx_emb = torch.stack([
                torch.tensor(self.retriever.embed_model.encode(c).mean(0)) 
                for c in ctx
            ]).to(device)
        ctx_emb = self.ctx_proj(ctx_emb)
        
        # Embeddings
        tok_emb = self.tok_emb(idx)
        pos = torch.cat([
            torch.zeros(B,1,device=device),
            torch.arange(1,T+1,device=device).repeat(B,1)
        ], dim=1)
        x = tok_emb + self.pos_emb(pos.long())
        x = torch.cat([ctx_emb.unsqueeze(1), x], dim=1)
        
        # Transformer
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        
        # Loss
        if targets is not None:
            targets = F.pad(targets, (1,0), value=-100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100
            )
        else:
            loss = None
            
        return logits, loss

    def generate(self, prompt, max_new_tokens=100, k=3):
        self.eval()
        with torch.no_grad():
            # Retrieve context
            ctx = self.retriever.retrieve([prompt], k=k)[0]
            ctx_emb = self.ctx_proj(
                torch.tensor(self.retriever.embed_model.encode(ctx))
                .mean(0).unsqueeze(0).to(device)
            )
            
            # Generate
            idx = torch.tensor(encode(prompt), device=device).unsqueeze(0)
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -block_size:])
                logits = logits[:, -1, :]
                idx = torch.cat([idx, logits.argmax(-1, keepdim=True)], -1)
                
        return decode(idx[0].tolist())

# ---------------------- OPTIMIZED TRAINING ----------------------
def main():
    # Data
    tokenizer = ByteLevelBPETokenizer.from_file("hindi_bpe/vocab.json")
    encode = lambda s: tokenizer.encode(s).ids
    decode = lambda l: tokenizer.decode(l)
    
    # Retriever
    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2').to(device)
    with open("hindi_fullarticles.json") as f:
        docs = [a["content"] for a in json.load(f) if "content" in a]
    retriever = FAISSRetriever(docs, embed_model)
    
    # Model
    model = RAGGPT(tokenizer.get_vocab_size(), retriever).to(device)
    model = torch.compile(model, mode='max-autotune')
    optimizer = Lion(model.parameters(), lr=learning_rate, use_fused=True)
    scaler = torch.cuda.amp.GradScaler()
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(
        BinaryDataset('train.bin'),
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=4
    )
    
    # Training loop
    for iter, (xb, yb) in enumerate(train_loader):
        if iter >= max_iters: break
            
        # Learning rate warmup
        lr = min(learning_rate, 1e-5 + (learning_rate-1e-5)*iter/warmup_iters)
        optimizer.param_groups[0]['lr'] = lr
        
        # Mixed precision
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        
        # Optimize
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Log
        if iter % 10 == 0:
            print(f"Iter {iter:4d} | Loss {loss.item():.4f} | LR {lr:.2e}")

    torch.save(model.state_dict(), 'rag_gpt_hindi_400m.pth')

if __name__ == "__main__":
    main()