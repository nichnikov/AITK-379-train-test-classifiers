import os
import json
import pickle 
import pandas as pd
import torch
from config import PATH
from texts_processing import TextsTokenizer
from transformers import AutoTokenizer, AutoModel

transformer_tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
transformer_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")


model_prefix = "20230913_1652_SysId10"
model_path = os.path.join(PATH, "models", model_prefix)
model_name = model_prefix + "_knn.model"
labels_file_name = model_prefix + "_labels.json"

with open(os.path.join(model_path, model_name), "rb") as m_f:
   model = pickle.load(m_f)

with open(os.path.join(model_path, labels_file_name), "r") as l_f:
                labels_dict_list = json.load(l_f)

labels_dict = {d["label"]: d["answer_id"] for d in labels_dict_list}

tokenizer = TextsTokenizer()
dataset_df = pd.read_csv(os.path.join(PATH, "data", "test_dataset_lb2int.csv"), sep="\t")

test_dataset = list(dataset_df.itertuples(index=False))

test_data = [(x.label, " ".join(tokenizer([str(x.text)])[0])) for x in test_dataset]

labels = pd.DataFrame(labels_dict_list)
result = []
true_pred = 0
not_other_pred = 0
for true_label, text in test_data:
    encoded_input = transformer_tokenizer([text], padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():     
        model_output = transformer_model(**encoded_input)
    embeddings = model_output.pooler_output

    predict_labels = model.predict_proba(embeddings)
    ans_scs = sorted([x for x in zip(labels["answer_id"], predict_labels[0]) if x[1] > 0.0],
                                 key=lambda x: x[1], reverse=True)
    ans_scs_tuples = [(x[0], x[1]) for x in ans_scs]
    if ans_scs_tuples[0][0] == true_label:
        true_pred += 1
    if ans_scs_tuples[0][0] != 4:
        not_other_pred += 1
    print("true_pred:", true_pred, "not_other_pred:", not_other_pred)

print("true_pred:", true_pred, "precision:", true_pred/not_other_pred)
print("not_other_pred:", not_other_pred, "recall:", not_other_pred/len(test_data))