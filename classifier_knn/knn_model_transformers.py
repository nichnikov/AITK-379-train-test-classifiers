import os
import json
import pandas as pd
from config import PATH
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from texts_processing import TextsTokenizer
import pickle
import torch


def knn_model_creater(sys_id: int, neighbors: int, is_lematize: bool, input_df: pd.DataFrame,
                      tokenizer: TextsTokenizer, pubs_list: list, transformer_model, transformer_tokenizer):
    """"""
    
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    dir_name = "".join([date_time, "_SysId", str(sys_id)])
    models_path = os.path.join(PATH, "models")

    if not os.path.exists(os.path.join(models_path, dir_name)):
        print(os.path.join(models_path, dir_name))
        os.mkdir(os.path.join(models_path, dir_name))

    labels = sorted(set(input_df["ID"]))
    labels_dict = [{"label": y, "answer_id": i} for y, i in enumerate(labels)]
    labels_file_name = "".join([dir_name, "_labels.json"])
    
    with open(os.path.join(models_path, dir_name, labels_file_name), "w") as f:
        json.dump(labels_dict, f, ensure_ascii=False)

    labels_df = pd.DataFrame(labels_dict)
    df_test_lbs = pd.merge(input_df, labels_df, left_on="ID", right_on="answer_id")
    train_df = df_test_lbs[["Cluster", "label"]]
    
    if is_lematize:
        train_tuples = list(train_df.itertuples(index=False))
        train_texts = [" ".join(tokenizer([x.Cluster])[0]) for x in train_tuples]
    else:
        train_texts = list(train_df["Cluster"])

    encoded_input = transformer_tokenizer(train_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')

    with torch.no_grad():
        model_output = transformer_model(**encoded_input)
    embeddings = model_output.pooler_output

    X = torch.nn.functional.normalize(embeddings)
    y = [x.label for x in train_tuples]

    neigh = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", algorithm="brute")
    neigh.fit(X, y)

    model_name = "".join([date_time, "_SysId", str(sys_id), "_knn.model"])
    with open(os.path.join(models_path, dir_name, model_name), "wb") as knn_f:
        pickle.dump(neigh, knn_f)

    parameters = {
        "date_time": date_time,
        "model_name": model_name,
        "n_neighbors": neighbors,
        "lemitize_texs": is_lematize,
        "pubs": list(pubs_list)
    }

    parameters_file_name = "".join([date_time, "_SysId", str(sys_id), "_parameters.json"])
    with open(os.path.join(models_path, dir_name, parameters_file_name), "w") as p_f:
        json.dump(parameters, p_f)