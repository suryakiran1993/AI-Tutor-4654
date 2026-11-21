# Custom BPE using predefined corpus

# Trains a custom BPE tokenizer using words read from corpus.txt.

from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Find the corpus beside this script, regardless of where the command is run.
corpus_path = Path(__file__).with_name("corpus.txt")
with open(corpus_path, "r", encoding="utf-8") as file:
    words = file.read().lower().split()

# Calculate word frequencies
word_frequency = Counter(words)

print("========== WORD FREQUENCIES ==========")
for word, frequency in word_frequency.items():
    print(word, ":", frequency)

# Count original character pairs, weighted by each word's frequency.
# BPE training then repeatedly merges frequent pairs into larger tokens.
pair_frequency = Counter()

print("\n========== ADJACENT CHARACTER PAIRS ==========")

for word, frequency in word_frequency.items():
    characters = list(word)
    pairs = list(zip(characters, characters[1:]))

    print("\nWord:", word)
    print("Character Pairs:", pairs)

    for pair in pairs:
        pair_frequency[pair] += frequency

print("\n========== PAIR FREQUENCIES ==========")

for pair, frequency in pair_frequency.most_common():
    print(pair, ":", frequency)

# Repeat each word so the trainer sees its frequency in the training data.
corpus = []
for word, frequency in word_frequency.items():
    corpus.extend([word] * frequency)

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Configure BPE training
trainer = BpeTrainer(
    vocab_size=20,
    min_frequency=2,
    special_tokens=["[UNK]"]
)

# Train the tokenizer
tokenizer.train_from_iterator(corpus, trainer)

# Display learned vocabulary
print("\n========== LEARNED VOCABULARY ==========")

vocabulary = tokenizer.get_vocab()

for token, token_id in sorted(
    vocabulary.items(), key=lambda item: item[1]
):
    print(token, ":", token_id)

# Test the tokenizer
text = input("\nEnter a sentence: ")

output = tokenizer.encode(text)

print("\n========== TOKENIZATION RESULT ==========")
print("Original Text:", text)
print("BPE Tokens:", output.tokens)
print("Token IDs:", output.ids)
print("Total Tokens:", len(output.tokens))