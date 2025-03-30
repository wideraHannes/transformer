import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm
import yaml
from config.paths import (
    BEST_MODELS,
    EXPERIMENT_DATA,
    EXPERIMENT_STORAGE,
    GPT2_FROM_BPE,
    LORA_MODELS,
    MODEL_CONFIG,
)
from src.experiment.models.SoftPrompt import setup_soft_prompting_model
from src.experiment.data.synthetic_data import create_dataloader
from src.experiment.models.LoRa import setup_lora_model
from src.modelling.model.transformer import TransformerConfig, TransformerModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def load_base_model(base_model_path, config):
    """
    Load the base transformer model from a checkpoint.

    Args:
        base_model_path (str or Path): Path to the base model checkpoint.
        config (dict): Transformer configuration dictionary.

    Returns:
        TransformerModel: Loaded base model.
    """
    base_model = TransformerModel(config=TransformerConfig(**config))
    checkpoint = torch.load(
        base_model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    return base_model


def evaluate_model(model, test_loader, tokenizer, device):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Initialize task-specific metrics
    task_metrics = {
        "sort up": {"loss": 0, "bleu": [], "accuracy": 0, "count": 0},
        "sort down": {"loss": 0, "bleu": [], "accuracy": 0, "count": 0},
        "add 1": {"loss": 0, "bleu": [], "accuracy": 0, "count": 0},
        "add 2": {"loss": 0, "bleu": [], "accuracy": 0, "count": 0},
    }

    task_prefixes = {
        "sort up:": "sort up",
        "sort down:": "sort down",
        "add 1:": "add 1",
        "add 2:": "add 2",
    }

    # Initialize overall metrics
    overall_metrics = {"loss": 0, "bleu": [], "accuracy": 0, "count": 0}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            _inputs = batch["input"]
            source_ids = batch["input_ids"].to(device)
            source_mask = batch["input_mask"].to(device)
            target_ids = batch["target_input"].to(device)
            target_mask = batch["target_mask"].to(device)
            labels = batch["target_label"].to(device)

            # Forward pass
            output = model(source_ids, target_ids, source_mask, target_mask)
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))

            # Convert logits to predicted token IDs
            predicted_ids = output.argmax(dim=-1)

            # Decode predictions and references
            preds = []
            for pred_seq in predicted_ids:
                eos_indices = torch.nonzero(
                    pred_seq == tokenizer.eos_token_id, as_tuple=True
                )[0]
                if len(eos_indices) > 0:
                    eos_index = eos_indices[0].item()
                    pred_seq = pred_seq[: eos_index + 1]
                preds.append(
                    tokenizer.decode(pred_seq.tolist(), skip_special_tokens=True)
                )

            refs = [
                tokenizer.decode(ref_seq.tolist(), skip_special_tokens=True)
                for ref_seq in labels
            ]

            # Determine task type from input prefix
            for input_text, pred, ref, batch_loss in zip(
                _inputs, preds, refs, [loss.item()] * len(_inputs)
            ):
                for prefix, task in task_prefixes.items():
                    if input_text.startswith(prefix):
                        # Update task-specific metrics
                        task_metrics[task]["loss"] += batch_loss
                        smooth_fn = SmoothingFunction().method1
                        bleu_score = sentence_bleu(
                            [ref.split()], pred.split(), smoothing_function=smooth_fn
                        )
                        task_metrics[task]["bleu"].append(bleu_score)
                        task_metrics[task]["accuracy"] += int(pred == ref)
                        task_metrics[task]["count"] += 1

                        # Update overall metrics
                        overall_metrics["loss"] += batch_loss
                        overall_metrics["bleu"].append(bleu_score)
                        overall_metrics["accuracy"] += int(pred == ref)
                        overall_metrics["count"] += 1

    # Calculate averages for each task
    for task in task_metrics.keys():
        if task_metrics[task]["count"] > 0:
            task_metrics[task]["loss"] /= task_metrics[task]["count"]
            task_metrics[task]["bleu"] = sum(task_metrics[task]["bleu"]) / len(
                task_metrics[task]["bleu"]
            )
            task_metrics[task]["accuracy"] /= task_metrics[task]["count"]

    # Calculate overall averages
    if overall_metrics["count"] > 0:
        overall_metrics["loss"] /= overall_metrics["count"]
        overall_metrics["bleu"] = sum(overall_metrics["bleu"]) / len(
            overall_metrics["bleu"]
        )
        overall_metrics["accuracy"] /= overall_metrics["count"]

    # Print task-specific metrics
    for task, metrics in task_metrics.items():
        print(f"\nTask: {task}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  BLEU: {metrics['bleu']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"  Loss: {overall_metrics['loss']:.4f}")
    print(f"  BLEU: {overall_metrics['bleu']:.4f}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")

    return task_metrics, overall_metrics


def load_lora(base_model, rank, lora_model_path, device):
    print("Loading LoRA model")
    # Setup the LoRA model
    lora_model = setup_lora_model(base_model, rank=rank)

    # Load the fine-tuned LoRA weights
    lora_checkpoint = torch.load(lora_model_path, map_location=device)
    lora_model.load_state_dict(lora_checkpoint.state_dict())
    lora_model = lora_model.to(device)
    lora_model.eval()
    return lora_model


def load_soft_prompting(base_model, soft_prompting_model_path, device):
    print("Loading SoftPrompting model")
    prompt_length = 10
    embedding_dim = 128
    # Setup the SoftPrompting model
    soft_prompting_model = setup_soft_prompting_model(
        base_model, prompt_length, embedding_dim
    )

    # Load the fine-tuned SoftPrompting weights
    soft_prompting_checkpoint = torch.load(
        soft_prompting_model_path, map_location=device
    )
    soft_prompting_model.load_state_dict(soft_prompting_checkpoint.state_dict())
    soft_prompting_model = soft_prompting_model.to(device)
    soft_prompting_model.eval()
    return soft_prompting_model


# Main function to run evaluation
def main_evaluation(type_="multi_task", model_="LoRa", rank=4):
    base_model_path = BEST_MODELS / "best_model.pth"  # Path to base model
    finetuned_model_path = EXPERIMENT_STORAGE / model_ / type_ / "best_model.pth"
    test_file = EXPERIMENT_DATA / "multi_task" / "test_data.json"  # Path to test data
    batch_size = 32

    # Load model configuration
    with open(MODEL_CONFIG, "r") as f:
        # Loading base model configuration
        model_config = yaml.safe_load(f)

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

    # Create test DataLoader
    test_loader = create_dataloader(
        test_file,
        batch_size,
        tokenizer,
        shuffle=False,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the base model
    base_model = load_base_model(base_model_path, model_config)
    base_model = base_model.to(device)

    model = None
    if model_ == "LoRa":
        # Load the LoRA model
        model = load_lora(base_model, rank, finetuned_model_path, device)
    elif model_ == "soft_prompt":
        model = load_soft_prompting(base_model, finetuned_model_path, device)

    # Evaluate LoRA model
    task_metrics, overall_metrics = evaluate_model(
        model,
        test_loader,
        tokenizer,
        device,
    )
    # Save metrics to a file
    metrics_file = EXPERIMENT_STORAGE / model_ / type_ / "metrics.yaml"

    # Save metrics to a file
    with open(metrics_file, "w") as f:
        yaml.dump(
            {
                "task_metrics": task_metrics,
                "overall_metrics": overall_metrics,
            },
            f,
        )


if __name__ == "__main__":
    rank = 32
    main_evaluation(type_=f"r_{rank}", model_="LoRa", rank=rank)
