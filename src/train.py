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

    print(dataset["train"][0])

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory for model checkpoints
        evaluation_strategy="epoch",  # Evaluate each epoch
        learning_rate=2e-5,  # Learning rate
        per_device_train_batch_size=8,  # Batch size for training
        per_device_eval_batch_size=8,  # Batch size for evaluation
        num_train_epochs=3,  # Number of training epochs
        weight_decay=0.01,  # Strength of weight decay
    )

    # Function to compute the accuracy of our model
    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    results = trainer.evaluate(dataset["test"])
    print(results)
