import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import get_dataset_and_labels


def run(cfg):

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset, label_mapping = get_dataset_and_labels(cfg, tokenizer)
    label_scheme = list(label_mapping.keys())
    num_labels = len(label_scheme)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )

    print(dataset["train"][0])

    training_args = TrainingArguments(
        output_dir=f"./models/{cfg.model_name}",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        warmup_ratio=0.05,
        weight_decay=0.01,
        learning_rate=cfg.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        eval_accumulation_steps=8,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=2,
        tf32=True,
        group_by_length=True,
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=1)

        index_to_label = {index: label for label, index in label_mapping.items()}

        # Get unique label indices in the true labels of the evaluation set
        unique_labels = sorted(set(labels))

        # Map these indices back to their string representations
        target_names = [index_to_label[label_index] for label_index in unique_labels]

        # Generate classification report only for labels present in y_true of evaluation data
        f1 = f1_score(labels, predictions, average="weighted", labels=unique_labels)
        accuracy = accuracy_score(labels, predictions)
        if cfg.method == "test":
            report = classification_report(
                labels,
                predictions,
                target_names=target_names,
                labels=unique_labels,
                digits=4,
            )

            print(report)

        return {"accuracy": accuracy, "f1": f1}

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if cfg.method == "train":

        # Train the model
        trainer.train()

    # Predict
    cfg.method = "test"

    print("Evaluating on test set...")

    results = trainer.predict(dataset["test"])
    print(results.metrics)
