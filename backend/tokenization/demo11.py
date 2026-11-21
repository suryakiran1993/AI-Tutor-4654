# Custom BPE using predefined words and frequencies

# Trains a small BPE tokenizer on predefined words and their frequencies, then tests it.

from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Define words and their frequencies
word_frequency = {
    "play": 5,
    "playing": 3,
    "played": 2
}

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
training_words = []

for word, frequency in word_frequency.items():
    training_words.extend([word] * frequency)

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

# Configure BPE trainer
trainer = BpeTrainer(
    vocab_size=50,
    min_frequency=2,
    special_tokens=["[UNK]"]
)

# Train tokenizer
tokenizer.train_from_iterator(training_words, trainer)

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