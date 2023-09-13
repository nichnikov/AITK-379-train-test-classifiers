"""
"""
import os
import json
import numpy as np
from datasets import DatasetDict
from transformers import (BertTokenizer,
                          BertModelWithHeads,
                          TrainingArguments,
                          AdapterTrainer,
                          EvalPrediction)
from config import PATH
# import torch

model_name = "bert-base-multilingual-cased"
adapter_name = "classifier_adapter"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModelWithHeads.from_pretrained(model_name)



def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(
        batch["text"],
        # batch["answer"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


dataset = DatasetDict.load_from_disk(os.path.join(PATH, "data", "train.data"))
print(dataset)

dataset = dataset.map(encode_batch, batched=True)
model.add_adapter(adapter_name)

# Add a matching classification head
model.add_classification_head(adapter_name, num_labels=15)
# Activate the adapter
model.train_adapter(adapter_name)

with open(os.path.join(PATH, "data", "config.json"), "r") as f:
    prmtrs = json.load(f)

print(prmtrs)

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=prmtrs["epochs"],
    per_device_train_batch_size=prmtrs["batch_size"],
    per_device_eval_batch_size=prmtrs["batch_size"],
    logging_steps=prmtrs["logging_steps"],
    save_steps=prmtrs["save_steps"],
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    # eval_dataset=dataset["train"],
    compute_metrics=compute_accuracy,
)

trainer.train()
# trainer.evaluate()

model.save_adapter(os.path.join(PATH, "models", adapter_name), adapter_name)