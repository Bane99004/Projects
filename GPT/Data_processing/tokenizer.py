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