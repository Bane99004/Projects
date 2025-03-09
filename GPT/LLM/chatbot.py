import torch 
import torch.nn as nn
from torch.nn import functional as F 
import mmap
import random
import pickle
import argparse
import os
from tokenizers import Tokenizer, ByteLevelBPETokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 3e-5
eval_iters = 100
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2

# chars = ""
# with open('vocab.txt','r', encoding = 'utf-8') as f:
#   text = f.read()
#   chars = sorted(list(set(text)))
  
# vocab_size = len(chars)
# string_to_int = { ch:i for i,ch in enumerate(chars) }
# int_to_string = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [string_to_int[c] for c in s]
# decode = lambda l: ''.join([int_to_string[i] for i in l])
# # data = torch.tensor(encode(text), dtype = torch.long)

with open ("file.json", "r", encoding=utf-8) as f:
  data =  json.load(f)

corpus_file="corpus.txt"
with open("corpus.txt", "w", encoding = utf-8) as f:
  for article in data:
    title = article.get("title", "")
    content = article.get("content", "")
    f.write(title + " " + content + "\n")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files = [corpus_file], vocab_size = 30000, min_frequency = 2,  special_tokens = ["<s>", "<pad>", "<unk>", "<mask>"])    
os.makedirs("hindi_bpe", exist_ok = True)
tokenizer.save_model("hindi_bpe")

tokenizer = Tokenizer.from_file("hindi_bpe/vocab.json")
encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)#(B,T,hs)
    q = self.query(x)#(B,T,hs)
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #(B,T,hs) @ (B, hs, T) ->(B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.head = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.head], dim = -1)#h(x) means we r paasing input to each head and it is storing its output in list(concatenating)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd),
      nn.Dropout(dropout)
    )
  
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa=MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    y = self.sa(x)
    x = self.ln1(x+y)
    y = self.ffwd(x)
    x = self.ln2(x+y)
    return x

class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*(Block(n_embd, n_head = n_head) for _ in range(n_layer)))
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.apply(self._init_weights)

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, index, targets=None):
    B,T = index.shape
    tok_emb = self.token_embedding_table(index)
    pos_emb = self.position_embedding_table(torch.arange(T, device = device))
   
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)
    
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      
    return logits, loss
  
  with open("hindi_fullarticles.json", "r" , encoding=utf-8) as f:
    articles = f.read()
  docs = [article["content"] for article in articles if article.get("content")]
  
  def dummy_embed(text):
    return np.random.rand(768).astype('float32')
  
  doc_embedding = np.array([dummy_embed(doc) for doc in docs])
  embedding_dim = doc_embedding.shape[1]
  
  index = faiss.IndexFlatL2(embedding_dim)
  index.add(doc_embedding)
  
  def retrieve_context(query, k=5):
    query_embedding = dummy_embed(query).reshape(1,-1)
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [docs[i] for i in indices[0]]
    return "\n".join(retrieved_docs)
  
  
  def generate(self, index, query, max_new_tokens):
    context = retrieve_context(query, k)
    prompt = context + "\n" + query
    inputs_ids = torch.tensor(encode(prompt)).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
      #index_cond = index[:, -1, :]#extract the last token but its not isolated it keeps the in context with its previous words in order to predict next word
      logits, _ = self.forward(index_cond) 
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = -1) 
      next_token = torch.multinomial(probs, num_samples=1)
      input_ids = torch.cat((input_ids, next_token), dim=1)
    return decode(input_ids.squeeze().tolist())
model = GPTLanguageModel(vocab_size)
m = model.to(device)

while True:
  prompt = input("Prompt:\n")
  context = torch.tensor(encode(prompt), dtype =torch.long, device = device)
  generated_chars = decode(m.generate(context.unsqueeze(0),prompt, max_new_tokens = 150)[0].tolist())
  print(f"completeion:\n{generated_chars}")
  
    