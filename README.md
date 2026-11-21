This is AI Tutor Project

## Tokenization demos

Install the packages needed for demos 9-14:

```shell
python -m pip install transformers tokenizers sentencepiece
```

Run a demo from the project root, for example:

```shell
python backend\tokenization\demo10.py
```

- Demo 9 uses GPT-2 BPE; demo 10 uses BERT WordPiece. Try `tokenization`
  in demo 10 to see `##`, which marks a continuation of a word.
- Demos 11 and 12 train BPE tokenizers on sample words and `corpus.txt`,
  respectively. Their printed pair counts describe the original character pairs,
  before training merges them.
- Demo 13 uses T5 SentencePiece. Its `▁` marker indicates a word boundary.
- Demo 14 trains a SentencePiece Unigram tokenizer using `corpus.txt`. It saves
  `my_tokenizer.model` and `my_tokenizer.vocab` in the current working directory,
  overwriting files with those names.

The pretrained models in demos 9, 10, and 13 need an internet connection the
first time they are downloaded. The corpus for demos 12 and 14 is located next
to the scripts. A small corpus may produce unknown tokens for new input.
