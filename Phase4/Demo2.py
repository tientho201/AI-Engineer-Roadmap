from collections import Counter, defaultdict

def train_bpe(corpus: list[str], num_merges: int = 20):
    """BPE: lặp lại việc gộp cặp ký tự xuất hiện nhiều nhất thành 1 token mới."""
    vocab = Counter()
    for word in " ".join(corpus).split():
        vocab[" ".join(word) + " </w>"] += 1

    merges = []
    for _ in range(num_merges):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split() # tách từ thành các ký tự riêng biệt - Tạo list
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        # Gộp cặp tốt nhất trong toàn bộ vocab
        bigram = " ".join(best)
        vocab = {w.replace(bigram, "".join(best)): f for w, f in vocab.items()}
    return merges, vocab , pairs


corpus = ["thấp thỏm thấp thỏm", "thấp thoáng", "thỏm thỏt", "thấp thỏm lo lắng"]
merges, vocab, pairs = train_bpe(corpus, num_merges=10)
print("Các lần gộp:", merges[:5])
print("Vocab cuối:", list(vocab)[:5])
print("Pairs:", pairs)
