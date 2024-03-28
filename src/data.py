import sys
from collections import Counter
import csv

csv.field_size_limit(sys.maxsize)

from sklearn.model_selection import train_test_split
from datasets import Dataset


def get_dataset_and_labels(cfg, tokenizer):

    with open(f"data/papyri_{cfg.data}.tsv") as file:
        data = list(csv.reader(file, delimiter="\t"))[1:]

    data = [x for x in data if x[2].strip()]

    labels = list(sorted({sublist[2] for sublist in data}))
    label_mapping = {label: id for id, label in enumerate(labels)}
    print(labels)
    print(len(data))

    # Filter out labels with fewer than 2 instances
    label_counts = Counter(item[2] for item in data)
    data = [item for item in data if label_counts[item[2]] > 1]

    print(len(data))

    # Extract features (texts) and labels according to their positions
    texts = [[item[3]] for item in data]  # Extract only the text
    ex_labels = [item[2] for item in data]  # Extract labels

    # Split into train and temporary test (which will become dev and test), stratifying by ex_labels
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, ex_labels, test_size=0.4, stratify=ex_labels, random_state=42
    )

    # Before the second split, filter out classes with fewer than 2 instances in y_temp
    temp_label_counts = Counter(y_temp)
    X_temp_filtered = [
        X_temp[i] for i, label in enumerate(y_temp) if temp_label_counts[label] > 1
    ]
    y_temp_filtered = [label for label in y_temp if temp_label_counts[label] > 1]

    # Now, perform the second stratified split with the filtered temp sets
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp_filtered,
        y_temp_filtered,
        test_size=0.5,
        stratify=y_temp_filtered,
        random_state=cfg.seed,
    )

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    # Recombine the features and labels into the desired dictionary format
    train = (
        Dataset.from_list(
            [
                {"label": label_mapping[y_train[i]], "text": X_train[i][0]}
                for i in range(len(X_train))
            ]
        )
        .map(tokenize_function, batched=True)
        .shuffle(seed=cfg.seed)
    )
    dev = (
        Dataset.from_list(
            [
                {"label": label_mapping[y_dev[i]], "text": X_dev[i][0]}
                for i in range(len(X_dev))
            ]
        )
        .map(tokenize_function, batched=True)
        .shuffle(seed=cfg.seed)
    )
    test = (
        Dataset.from_list(
            [
                {"label": label_mapping[y_test[i]], "text": X_test[i][0]}
                for i in range(len(X_test))
            ]
        )
        .map(tokenize_function, batched=True)
        .shuffle(seed=cfg.seed)
    )

    print("Train:", len(train))
    print("Dev:", len(dev))
    print("Test:", len(test))

    print(sum([len(train), len(dev), len(test)]))

    return {"train": train, "dev": dev, "test": test}, labels
