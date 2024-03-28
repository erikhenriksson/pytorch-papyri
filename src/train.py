import glob
import json
import os
import random
import shutil
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from .data import get_dataset_and_labels

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


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

    num_labels = len(label_mapping)  # Make sure label_mapping is defined as before
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )

    label_names = {id: label for label, id in label_mapping.items()}

    print(dataset["train"][0])

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory for model checkpoints
        overwrite_output_dir=True,
        num_train_epochs=30,
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
        f1 = f1_score(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)
        # Use the inverted label mapping to provide label names for the classification report
        print(
            classification_report(
                labels,
                predictions,
                target_names=[label_names[id] for id in sorted(label_names)],
                digits=4,
            )
        )
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

    # Train the model
    trainer.train()

    results = trainer.predict(dataset["test"])
    print(results)
