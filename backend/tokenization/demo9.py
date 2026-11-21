# BPE Tokenization using Pretrained GPT-2
# GPT-2 uses a special marker for spaces before some tokens.

# Uses GPT-2’s pretrained BPE tokenizer to split text into tokens and IDs, count frequencies, and decode it.

from collections import Counter
from transformers import AutoTokenizer

# Load pretrained GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Get input sentence
text = input("Enter a sentence: ") or "I love Learning python"

# Generate BPE tokens
tokens = tokenizer.tokenize(text)

# Convert tokens into IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n========== TOKENIZATION RESULT ==========")

print("Original Sentence:", text)
print("BPE Tokens:", tokens)
print("Token IDs:", token_ids)
print("Total Tokens:", len(tokens))

# Calculate individual token frequencies
token_frequency = Counter(tokens)

print("\n========== TOKEN FREQUENCIES ==========")

for token, frequency in token_frequency.items():
    print(token, ":", frequency)

# Count neighboring tokens; tokens[1:] is the same list without its first token.
pair_frequency = Counter(zip(tokens, tokens[1:]))

print("\n========== ADJACENT TOKEN PAIR FREQUENCIES ==========")

for pair, frequency in pair_frequency.items():
    print(pair, ":", frequency)

# Decode tokens
decoded_text = tokenizer.decode(token_ids)

print("\nDecoded Text:", decoded_text)