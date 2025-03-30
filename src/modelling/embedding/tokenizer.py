from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import GPT2Tokenizer
import json

from config.paths import GPT2_FROM_BPE


class CustomBPETokenizer:
    """Create a custom tokenizer for natural language processing tasks
    Train a BPE tokenizer on a specific dataset
    Convert the trained tokenizer to a GPT-2 compatible format"""

    def __init__(self, dataset, vocab_size=50000, max_length=64):
        """
        Extracting text data from a given dataset
        Training a BPE tokenizer
        Converting the tokenizer to GPT-2 format
        """

        self.data = [
            translation
            for example in dataset
            for translation in (example["src"], example["tgt"])
        ]
        self.max_length = max_length
        self.bpe_tokenizer = self.train_bpe_tokenizer(vocab_size)
        self.gpt2_tokenizer = self.convert_to_gpt2_tokenizer()

    def train_bpe_tokenizer(self, vocab_size):
        """
        Configures BPE tokenization parameters
        Sets special tokens ([PAD], [BOS], [EOS], [UNK])
        Trains the tokenizer on the provided data
        """

        bpe_tokenizer = Tokenizer(models.BPE())

        bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        bpe_tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            special_tokens=[
                "[PAD]",
                "[BOS]",
                "[EOS]",
                "[UNK]",
            ],
            vocab_size=vocab_size,
            show_progress=True,
        )

        bpe_tokenizer.train_from_iterator(self.data, trainer=trainer)

        return bpe_tokenizer

    def convert_to_gpt2_tokenizer(self):
        """
        Extracts vocabulary and merge rules from trained BPE tokenizer
        Saves vocabulary and merge information to files
        Creates a GPT-2 compatible tokenizer
        """
        model = json.loads(self.bpe_tokenizer.to_str())["model"]
        vocab_dict = model["vocab"]
        merges_list = model["merges"]

        vocab_path = str(GPT2_FROM_BPE / str(self.max_length) / "vocab.json")
        merges_path = str(GPT2_FROM_BPE / str(self.max_length) / "merges.txt")
        # Save vocab.json and merges.txt

        with open(vocab_path, "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
        with open(merges_path, "w") as merges_file:
            merges_file.write(
                "\n".join(" ".join(map(str, merge)) for merge in merges_list)
            )

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(str(GPT2_FROM_BPE))
        # Explicitly set special tokens
        gpt2_tokenizer.pad_token = "[PAD]"
        gpt2_tokenizer.bos_token = "[BOS]"
        gpt2_tokenizer.eos_token = "[EOS]"
        gpt2_tokenizer.unk_token = "[UNK]"

        # Update the tokenizer's vocabulary
        gpt2_tokenizer.add_special_tokens(
            {
                "pad_token": "[PAD]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
            }
        )

        return gpt2_tokenizer

    def convert_tokens_to_ids(self, token):
        return self.gpt2_tokenizer.convert_tokens_to_ids(token)

    def tokenize(self, example):
        return self.gpt2_tokenizer.tokenize(example)

    def encode(self, example):
        return self.gpt2_tokenizer.encode(example, add_special_tokens=True)

    def decode(self, tokens):
        return self.gpt2_tokenizer.decode(tokens, skip_special_tokens=True)
