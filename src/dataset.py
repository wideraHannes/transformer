import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer

from config.paths import GPT2_FROM_BPE
from src.utils.data_cleaning import clean_dataset


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Store special token IDs as class attributes
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def add_padding_or_truncate(self, token_ids, has_bos=False, has_eos=False):
        """
        Pad or truncate a list of token IDs to max_length.
        If is_target is True, only add BOS at the beginning (no EOS token).
        Otherwise, add both BOS and EOS tokens.
        """

        effective_length = (
            self.max_length - 2 if has_bos and has_eos else self.max_length - 1
        )

        # Truncate the sequence if it's longer than the effective length
        if len(token_ids) > effective_length:
            token_ids = token_ids[-effective_length:]

        # Add BOS token at the beginning
        if has_bos:
            token_ids = [self.bos_token_id] + token_ids

        if has_eos:
            token_ids = token_ids + [self.eos_token_id]

        # Pad the sequence to max_length if necessary
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length

        return token_ids

    def create_attention_mask(self, token_ids):
        """Create attention mask from token IDs"""
        return [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]

    def check_format(self, source_ids, target_input_ids, target_labels):
        # Assert sequence lengths
        assert len(source_ids) == self.max_length, "Source sequence length mismatch"
        assert (
            len(target_input_ids) == self.max_length
        ), "Target input sequence length mismatch"
        assert (
            len(target_labels) == self.max_length
        ), "Target labels sequence length mismatch"

        # Assert `[BOS]` and `[EOS]` positions
        assert (
            source_ids[0] == self.bos_token_id
        ), "Source sequence must start with [BOS]"
        assert (
            target_input_ids[0] == self.bos_token_id
        ), "Target input must start with [BOS]"
        assert self.eos_token_id in source_ids, "[EOS] must exist in source sequence"
        assert self.eos_token_id in target_labels, "[EOS] must exist in target labels"

        # Check `[EOS]` position and `[PAD]` padding after `[EOS]`
        source_eos_index = source_ids.index(self.eos_token_id)
        target_labels_eos_index = target_labels.index(self.eos_token_id)

        assert all(
            token == self.pad_token_id for token in source_ids[source_eos_index + 1 :]
        ), "Source sequence should only contain [PAD] after [EOS]"
        assert all(
            token == self.pad_token_id
            for token in target_labels[target_labels_eos_index + 1 :]
        ), "Target labels should only contain [PAD] after [EOS]"

    def __getitem__(self, idx):
        source = self.data[idx]["src"]
        target = self.data[idx]["tgt"]
        # Encode source and target texts
        encoded_source = self.tokenizer.encode(
            source, add_special_tokens=False  # We'll add special tokens manually
        )
        encoded_target = self.tokenizer.encode(target, add_special_tokens=False)

        # Process source sequence
        source_ids = self.add_padding_or_truncate(
            encoded_source, has_bos=True, has_eos=True
        )
        target_input_ids = self.add_padding_or_truncate(encoded_target, has_bos=True)
        target_labels = self.add_padding_or_truncate(encoded_target, has_eos=True)

        source_mask = self.create_attention_mask(source_ids)
        target_mask = self.create_attention_mask(target_input_ids)

        # Check the format of the processed sequences
        # self.check_format(source_ids, target_input_ids, target_labels)

        return {
            "source": source,
            "target": target,
            "source_ids": torch.tensor(source_ids),  # bos tokens eos
            "source_mask": torch.tensor(source_mask),
            "target_ids": torch.tensor(target_input_ids),  # bos tokens
            "target_mask": torch.tensor(target_mask),
            "labels": torch.tensor(target_labels),  # tokens eos
        }


def main():
    # Load and prepare datasets
    train_ds = load_dataset("wmt17", "de-en", split="train[:50]")
    max_length = 16
    train_cleaned = clean_dataset(train_ds, max_length)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(str(GPT2_FROM_BPE / str(max_length)))
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
        }
    )

    # Create dataset
    train_dataset = TranslationDataset(train_cleaned, tokenizer, max_length=max_length)
    # Print 5 examples from the training dataset
    print("\n5 examples from the training dataset:")
    for i in range(2):
        example = train_dataset[i]
        print(f"Example {i}:")
        print(f"Source IDs: {example['source_ids']}")
        print(f"Target IDs: {example['target_ids']}")
        print(f"Label Ids: {example['labels']}")
        print(f"Source Mask: {example['source_mask']}")
        print(f"Target Mask: {example['target_mask']}")
        print()


if __name__ == "__main__":
    main()
