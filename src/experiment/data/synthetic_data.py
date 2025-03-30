import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

from config.paths import EXPERIMENT_DATA, GPT2_FROM_BPE


class SortingDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=64):
        """
        Custom Dataset to load and tokenize sorting data from a JSON file.

        Args:
            json_file (str): Path to the JSON file containing the dataset.
            tokenizer (GPT2Tokenizer): The tokenizer to process sequences.
            max_length (int): Maximum sequence length for padding/truncation.
            type (str): The type of target sequence to return.
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)
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
        If has_bos or has_eos is True, special tokens are added.
        """
        effective_length = (
            self.max_length - 2 if has_bos and has_eos else self.max_length - 1
        )

        # Truncate the sequence if it's longer than the effective length
        if len(token_ids) > effective_length:
            token_ids = token_ids[:effective_length]

        # Add BOS token at the beginning
        if has_bos:
            token_ids = [self.bos_token_id] + token_ids

        # Add EOS token at the end
        if has_eos:
            token_ids = token_ids + [self.eos_token_id]

        # Pad the sequence to max_length if necessary
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length

        return token_ids

    def create_attention_mask(self, token_ids):
        """Create attention mask from token IDs."""
        return [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]

    def process_target(self, encoded_target):
        """
        Process a target sequence into shifted input and label target.
        The input contains [BOS] and tokens, while the label contains tokens and [EOS].

        Args:
            encoded_target (list): Tokenized target sequence.

        Returns:
            tuple: Target input and label target.
        """
        target_input = self.add_padding_or_truncate(encoded_target, has_bos=True)
        target_label = self.add_padding_or_truncate(encoded_target, has_eos=True)
        return target_input, target_label

    def __getitem__(self, idx):
        """
        Retrieve and tokenize a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Tokenized inputs and a specific target type with attention mask.
        """
        item = self.data[idx]

        # Tokenize the input sequence
        encoded_input = self.tokenizer.encode(
            " ".join(map(str, item["input"])), add_special_tokens=False
        )
        input_ids = self.add_padding_or_truncate(
            encoded_input, has_bos=True, has_eos=True
        )
        input_mask = self.create_attention_mask(input_ids)

        # Tokenize and process the specific target type
        encoded_target = self.tokenizer.encode(
            " ".join(map(str, item["target"])), add_special_tokens=False
        )
        target_input, target_label = self.process_target(encoded_target)
        target_input_mask = self.create_attention_mask(target_input)

        return {
            "input": item["input"],
            "target": item["target"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "target_input": torch.tensor(target_input, dtype=torch.long),
            "target_mask": torch.tensor(target_input_mask, dtype=torch.long),
            "target_label": torch.tensor(target_label, dtype=torch.long),
        }


# Function to create DataLoader
def create_dataloader(json_file, batch_size, tokenizer, max_length=64, shuffle=True):
    """
    Create a PyTorch DataLoader for the sorting dataset.

    Args:
        json_file (str): Path to the JSON file containing the dataset.
        batch_size (int): Number of samples per batch.
        tokenizer (GPT2Tokenizer): Tokenizer for processing sequences.
        max_length (int): Maximum sequence length for padding/truncation.
        shuffle (bool): Whether to shuffle the data.
        type (str): The type of target sequence to return.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    dataset = SortingDataset(json_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# Example usage
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(str(GPT2_FROM_BPE / "64"))
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
        }
    )

    # File paths
    train_file = EXPERIMENT_DATA / "multi_task" / "train_data.json"
    val_file = EXPERIMENT_DATA / "multi_task" / "val_data.json"
    test_file = EXPERIMENT_DATA / "multi_task" / "test_data.json"

    # Create DataLoaders
    batch_size = 8
    max_length = 10

    train_loader = create_dataloader(
        train_file, batch_size, tokenizer, max_length, shuffle=True
    )

    # Inspect a batch
    for batch in train_loader:
        print("Input Sequences:", batch["input"])
        print("Target Sequences:", batch["target"])
        print("Input IDs:", batch["input_ids"])
        print("Input Mask:", batch["input_mask"])
        print("Target Input:", batch["target_input"])
        print("Target Label:", batch["target_label"])
        print("Target Mask:", batch["target_mask"])
        break
