import numpy as np
import json
from sklearn.model_selection import train_test_split
from config.paths import EXPERIMENT_DATA


def generate_sorting_data(num_samples, seq_length, min_val, max_val):
    data = []
    task_prefixes = {
        "target_increasing": "sort up:",
        "target_decreasing": "sort down:",
        "target_plus_one": "add 1:",
        "target_plus_two": "add 2:",
    }

    for _ in range(num_samples):
        seq = np.random.randint(min_val, max_val, seq_length).tolist()
        targets = {
            "target_increasing": sorted(seq),
            "target_decreasing": sorted(seq, reverse=True),
            "target_plus_one": [x + 1 for x in seq],  # Preserve the original order
            "target_plus_two": [x + 2 for x in seq],  # Preserve the original order
        }

        # Create one entry per task
        for task_key, prefix in task_prefixes.items():
            input_str = f"{prefix} {' '.join(map(str, seq))}"
            data.append({"input": input_str, "target": targets[task_key]})

    return data


def save_to_json(data, filename):
    """
    Saves the data to a JSON file.

    Args:
        data (list): The dataset containing input sequences and targets.
        filename (str): Path to save the JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} samples to {filename}")


# Parameters
total_samples = 500  # Set your desired number of samples
seq_length = 4
min_val = 1
max_val = 100

# Generate the full dataset
full_data = generate_sorting_data(total_samples, seq_length, min_val, max_val)

# Split into train (80%), validation (10%), and test (10%)
train_data, temp_data = train_test_split(full_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save splits to JSON files
save_to_json(train_data, EXPERIMENT_DATA / "small_m" / "train_data.json")
save_to_json(val_data, EXPERIMENT_DATA / "small_m" / "val_data.json")
save_to_json(test_data, EXPERIMENT_DATA / "small_m" / "test_data.json")
