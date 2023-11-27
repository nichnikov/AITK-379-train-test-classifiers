import os
import numpy as np
import pandas as pd
from datasets import DatasetDict
from transformers import (BertTokenizer, 
                          BertModelWithHeads,
                          )
from transformers.adapters.composition import Fuse
from texts_processing import TextsTokenizer
import torch



def predict(hypothesis):
  encoded = tokenizer(hypothesis, max_length=512, return_tensors="pt")
  # if torch.cuda.is_available():
  #  encoded.to("cuda")
  logits = model(**encoded)[0]
  tanh = torch.tanh(logits)
  pred_class = torch.argmax(logits).item()
  # print("sigmoid:", torch.sigmoid(logits))
  return pred_class



lematizer = TextsTokenizer()
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)


adapter_name = "classifier_adapter"

# adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path = os.path.join(os.getcwd(), "models", adapter_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)

train_df = pd.read_csv(os.path.join("data", "train_dataset_lb2int.csv"), sep="\t")
lbs_dict = {lb: tg for lb, tg in  set(zip(train_df["label"], train_df["label_str"]))}
print(lbs_dict)

dataset_df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_dataset_lb2int_full.csv"), sep="\t")

test_dataset = list(dataset_df.itertuples(index=False))
test_data = [(x.label, " ".join(lematizer([str(x.text)])[0]), x.label_str) for x in test_dataset]



results = []
true_pred = 0
not_other_pred = 0
k = 1
for true_label, text, tag in test_data:
    prd_label = predict(text)
    results.append({"Query": text, 
                    "predict_lb": prd_label,
                    "predict_tag": lbs_dict[prd_label],
                    "true_lb": true_label,
                    "true_tag": tag})
    print(k, "/", len(test_data), "predict:", prd_label, "true:", true_label)
    k += 1

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(os.path.join(os.getcwd(), "results", "results_bert_adapter_classifier.csv"), sep="\t", index=False)
