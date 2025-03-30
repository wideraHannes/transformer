from datetime import datetime
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config.paths import GPT2_FROM_BPE, HYPERPARAMETERS, MODEL_CONFIG, BEST_MODELS
from src.modelling.model.transformer import TransformerConfig, TransformerModel
from src.utils.data_cleaning import clean_dataset
from transformers import GPT2Tokenizer
from src.dataset import TranslationDataset
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml
from tqdm import tqdm
import logging
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.utils.lr_scheduler import TransformerLRScheduler


class TrainerConfig:
    def __init__(self, **kwargs):
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)

        # train_subset: ":10000" , warmupsteps are 5-10% of total steps

        self.val_interval = kwargs.get("val_interval", 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps:0")
        print(f"Using device: {self.device}")


class TransformerTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        config: TrainerConfig,
        model_config: dict,
    ):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.model_config = model_config

        # Initialize dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True
        )

        train_subset = len(train_dataset)
        total_steps = (train_subset // config.batch_size) * config.num_epochs
        self.warmup_steps = min(int(total_steps * 0.1), 35000)
        print(f"Total steps: {total_steps}, Warmup steps: {self.warmup_steps}")

        # Initialize optimizer and scheduler
        # parameters from the paper
        self.optimizer = AdamW(
            model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9
        )
        print(self.model_config)
        self.scheduler = TransformerLRScheduler(
            self.optimizer, self.model_config["d_model"], self.warmup_steps
        )

        # Initialize loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
        )

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize WandB
        wandb.init(project="transformer_experiment", config=config.__dict__)
        wandb.config.update(model_config)
        wandb.define_metric("Train", step_metric="epoch")
        wandb.define_metric("Validation", step_metric="epoch")
        wandb.define_metric("Learning Rate", step_metric="epoch")
        wandb.define_metric("BLEU", step_metric="epoch")

    def train_epoch(self, epoch):
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training")
        for batch in progress_bar:
            # Move batch to device
            source_ids = batch["source_ids"].to(self.config.device)
            source_mask = batch["source_mask"].to(self.config.device)
            target_ids = batch["target_ids"].to(self.config.device)
            target_mask = batch["target_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(source_ids, target_ids, source_mask, target_mask)
            logits = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            # Calculate loss and backward pass
            loss = self.loss_fn(logits, labels)
            loss.backward()

            # Optimize
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        wandb.log({"Train": avg_loss, "epoch": epoch + 1})
        return avg_loss

    def calculate_bleu_from_logits(self, logits, labels):
        """
        Calculate the BLEU score for a batch of logits and labels.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): Ground-truth labels of shape (batch_size, seq_len).

        Returns:
            float: Average BLEU score for the batch.
        """
        # for first sample print probability of highest eos
        softmax_probs = F.softmax(
            logits[0], dim=-1
        )  # Apply softmax over vocab dimension

        # Extract probabilities for [EOS] and [PAD] tokens
        eos_probs = softmax_probs[:, 2]  # Probabilities for [EOS]
        # print highest probability for eos
        self.logger.info(f"highest probability for [EOS]: {eos_probs.max().item()}")

        # Convert logits to predicted token IDs
        predicted_ids = logits.argmax(dim=-1)  # Shape: (batch_size, seq_len)

        preds = []
        for pred_seq in predicted_ids:
            # Find the index of [EOS] in pred_seq
            eos_indices = torch.nonzero(
                pred_seq == self.tokenizer.eos_token_id, as_tuple=True
            )[0]
            if len(eos_indices) > 0:
                eos_index = eos_indices[0].item()  # First occurrence of [EOS]
                pred_seq = pred_seq[: eos_index + 1]  # Include [EOS] itself

            # Decode the sequence
            preds.append(
                self.tokenizer.decode(pred_seq.tolist(), skip_special_tokens=True)
            )

        refs = [
            self.tokenizer.decode(ref_seq.tolist(), skip_special_tokens=True)
            for ref_seq in labels
        ]

        # Log one example during validation
        self.logger.info(
            f"Example from validation set: \nPredicted: {preds[0]} \nReference: {refs[0]}"
        )

        # Calculate BLEU score
        smooth_fn = SmoothingFunction().method1
        batch_bleu = [
            sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
            for pred, ref in zip(preds, refs)
        ]
        return sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0.0

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        all_bleu_scores = []

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} Validating")
        for batch in progress_bar:
            # Move batch to device
            source_ids = batch["source_ids"].to(self.config.device)
            source_mask = batch["source_mask"].to(self.config.device)
            target_ids = batch["target_ids"].to(self.config.device)
            target_mask = batch["target_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # Forward pass
            output = self.model(source_ids, target_ids, source_mask, target_mask)

            bleu_score = self.calculate_bleu_from_logits(output, labels)
            all_bleu_scores.append(bleu_score)

            logits = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            # Calculate loss
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()

            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.val_loader)
        avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
        wandb.log({"Validation": avg_loss, "epoch": epoch + 1})
        wandb.log({"BLEU": avg_bleu, "epoch": epoch + 1})
        return avg_loss

    def train(self):
        """Main training loop."""
        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            # log current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({"Learning Rate": current_lr, "epoch": epoch})
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Training phase
            train_loss = self.train_epoch(epoch)
            self.logger.info(f"Training loss: {train_loss:.4f}")

            # Validation phase
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate(epoch)
                self.logger.info(f"Validation loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(BEST_MODELS / "best_model.pth")
                    self.logger.info("New best model saved!")

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": self.model_config,
        }
        torch.save(checkpoint, filename)


def main():
    # Load configurations
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)
    with open(HYPERPARAMETERS, "r") as f:
        hparams = yaml.safe_load(f)

    # Create trainer config
    trainer_config = TrainerConfig(**hparams)

    # Load and prepare datasets
    train_ds = load_dataset("wmt17", "de-en", split=f"train[{hparams['train_subset']}]")
    val_ds = load_dataset(
        "wmt17", "de-en", split=f"validation[{hparams["val_subset"]}]"
    )

    max_length = model_config.get("max_len", 64)
    train_cleaned = clean_dataset(train_ds, max_length)
    val_cleaned = clean_dataset(val_ds, max_length)
    print("train dataset size: ", len(train_cleaned))
    print("val dataset size: ", len(val_cleaned))

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(str(GPT2_FROM_BPE / str(max_length)))
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",  # 0
            "bos_token": "[BOS]",  # 1
            "eos_token": "[EOS]",  # 2
            "unk_token": "[UNK]",  # 3
        }
    )

    # Create datasets
    train_dataset = TranslationDataset(train_cleaned, tokenizer, max_length=max_length)
    val_dataset = TranslationDataset(val_cleaned, tokenizer, max_length=max_length)

    # Initialize model
    transformer_config = TransformerConfig(**model_config)
    model = TransformerModel(config=transformer_config)

    # Create trainer and start training
    trainer = TransformerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        model_config=model_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
