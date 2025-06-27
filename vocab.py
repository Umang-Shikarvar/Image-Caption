import os
import pickle
from collections import Counter

import nltk
from pycocotools.coco import COCO

nltk.download('punkt_tab')                                     # Download the punkt tokenizer

# Smaple tokenizing
sample="This is a sample caption"
sample_tokens = nltk.tokenize.word_tokenize(sample.lower())
print(f"Sample tokens: {sample_tokens}")

# CONFIG
VOCAB_THRESHOLD = 5                                         # Minimum word frequency to include in vocabulary
ANNOTATIONS_FILE = "annotations/captions_train2017.json"    # Path to COCO annotations file
VOCAB_PKL_PATH = "vocab.pkl"                                # Path to save vocabulary as a pickle file              
VOCAB_TXT_PATH = "vocab.txt"                                # Path to save vocabulary as a text file
SPECIAL_TOKENS = ["<start>", "<end>", "<unk>"]              # Special tokens for start, end, and unknown words

# Load COCO
print("Loading COCO annotations...")
coco = COCO(ANNOTATIONS_FILE)

# Tokenize
print("Tokenizing captions...")
counter = Counter()
ids = coco.anns.keys()
for i, idx in enumerate(ids):                               # Iterate over all annotation IDs
    caption = str(coco.anns[idx]["caption"]).lower()
    tokens = nltk.tokenize.word_tokenize(caption)
    counter.update(tokens)
    if i % 100000 == 0:
        print(f"[{i}/{len(ids)}] Tokenizing...")

# Build vocab
print("Building vocabulary...")
words = [word for word, cnt in counter.items() if cnt >= VOCAB_THRESHOLD]
word2idx = {}
idx2word = {}
idx = 0

for token in SPECIAL_TOKENS:
    word2idx[token] = idx
    idx2word[idx] = token
    idx += 1

for word in words:
    if word not in word2idx:
        word2idx[word] = idx
        idx2word[idx] = word
        idx += 1

print(f"Total vocab size: {len(word2idx)}")

# Save vocab
with open(VOCAB_PKL_PATH, "wb") as f:
    pickle.dump({"word2idx": word2idx, "idx2word": idx2word}, f)
print(f"Saved vocab to {VOCAB_PKL_PATH}")

with open(VOCAB_TXT_PATH, "w") as f:
    for idx in range(len(idx2word)):
        f.write(f"{idx}\t{idx2word[idx]}\n")
print(f"Saved vocab text to {VOCAB_TXT_PATH}")