import os
import json
import pandas as pd
from config import PATH
from embeddings import Embedding, transformer_func
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from texts_processing import TextsTokenizer
import pickle
from datetime import datetime
# from src.config import logger


def knn_model_creater(sys_id: int, neighbors: int, is_lematize: bool, input_df: pd.DataFrame,
                      tokenizer: TextsTokenizer, pubs_list: list):
    """"""
    transformer_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embedder = Embedding(transformer_model, transformer_func)
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
        json.dump(labels_dict, f)

    labels_df = pd.DataFrame(labels_dict)
    df_test_lbs = pd.merge(input_df, labels_df, left_on="ID", right_on="answer_id")
    train_df = df_test_lbs[["Cluster", "label"]]
    train_df = train_df.sample(frac=1)
    if is_lematize:
        lm_texts = tokenizer(list(train_df["Cluster"]))
        train_texts = [" ".join(lm_tx) for lm_tx in lm_texts]
    else:
        train_texts = list(train_df["Cluster"])

    X = embedder(train_texts)
    y = list(train_df["label"])

    # logger.info("SysId: {}, X.shape:, {}, len(y): {}".format(str(sys_id), str(X.shape), str(len(y))))

    neigh = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", algorithm="brute")
    neigh.fit(X, y)

    model_name = "".join([date_time, "_SysId", str(sys_id), "_knn.model"])
    with open(os.path.join(models_path, dir_name, model_name), "wb") as knn_f:
        pickle.dump(neigh, knn_f)

    # logger.info("I have done and saved model: ".format(str(model_name)))

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