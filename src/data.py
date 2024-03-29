import csv
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

from datasets import Dataset
from sklearn.model_selection import train_test_split


def get_dataset_and_labels(cfg, tokenizer):
    with open(f"data/papyri_{cfg.data}.tsv") as file:
        data = list(csv.reader(file, delimiter="\t"))[1:]

    data = [x for x in data if x[2].strip()]
    print(f"Initial data size: {len(data)}")

    # Filter out labels with fewer than 2 instances initially
    label_counts = Counter(item[2] for item in data)
    data = [item for item in data if label_counts[item[2]] >= 10]
    print(f"Filtered data size (examples >= 10): {len(data)}")

    data = [
        item
        for item in data
        if item[2].startswith("Egypt")
        and not ["Upper"] in item[2]
        and not ["Coast"] in item[2]
        and not ["Thebais"] in item[2]
    ]
    print(f"Filtered data size (Egypt): {len(data)}")
    # Extract features (texts) and labels according to their positions
    texts = [[item[3]] for item in data]
    ex_labels = [item[2] for item in data]

    # First split into train and temporary test (to become dev and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, ex_labels, test_size=0.2, stratify=ex_labels, random_state=42
    )

    # Before the second split, filter out classes with fewer than 2 instances in y_temp again if needed
    temp_label_counts = Counter(y_temp)
    X_temp_filtered = [
        X_temp[i] for i, label in enumerate(y_temp) if temp_label_counts[label] >= 1
    ]
    y_temp_filtered = [label for label in y_temp if temp_label_counts[label] >= 1]

    # Second split into dev and test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp_filtered,
        y_temp_filtered,
        test_size=0.5,
        stratify=y_temp_filtered,
        random_state=cfg.seed,
    )

    # Update labels and label_mapping after filtering
    final_labels = sorted(set(y_train + y_dev + y_test))
    label_mapping = {label: id for id, label in enumerate(final_labels)}

    print(f"Final labels (after filtering): {final_labels}")

    # Tokenization and conversion to dataset
    def tokenize_and_create_dataset(X, y):
        tokenized_texts = tokenizer(
            [x[0] for x in X],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        labels = [label_mapping[label] for label in y]
        return Dataset.from_dict(
            {
                "input_ids": tokenized_texts["input_ids"],
                "attention_mask": tokenized_texts["attention_mask"],
                "labels": labels,
            }
        )

    train_dataset = tokenize_and_create_dataset(X_train, y_train).shuffle(seed=cfg.seed)
    dev_dataset = tokenize_and_create_dataset(X_dev, y_dev).shuffle(seed=cfg.seed)
    test_dataset = tokenize_and_create_dataset(X_test, y_test).shuffle(seed=cfg.seed)

    print(
        f"Train size: {len(train_dataset)}, Dev size: {len(dev_dataset)}, Test size: {len(test_dataset)}"
    )

    return (
        {"train": train_dataset, "dev": dev_dataset, "test": test_dataset},
        label_mapping,
    )
