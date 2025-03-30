from datasets import load_dataset
import yaml
from config.paths import MODEL_CONFIG
from src.utils.data_cleaning import clean_dataset
from src.modelling.embedding.tokenizer import CustomBPETokenizer

# @TODO do with more percentage


#

with open(MODEL_CONFIG, "r") as f:
    model_config = yaml.safe_load(f)


# Load the dataset
def init_tokenizer(percentage=1):
    dataset = load_dataset("wmt17", "de-en", split=f"train[:{percentage}%]")
    cleaned_data = clean_dataset(dataset, max_length=model_config["max_len"])
    CustomBPETokenizer(
        cleaned_data,
        vocab_size=model_config["vocab_size"],
        max_length=model_config["max_len"],
    )


init_tokenizer(100)
