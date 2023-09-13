import os
import json
import pickle 
from embeddings import (Embedding, transformer_func)
import pandas as pd
from config import PATH
from sentence_transformers import SentenceTransformer
from texts_processing import TextsTokenizer


model_prefix = "20230913_1345_SysId10"
model_path = os.path.join(PATH, "models", model_prefix)
model_name = model_prefix + "_knn.model"
labels_file_name = model_prefix + "_labels.json"

with open(os.path.join(model_path, model_name), "rb") as m_f:
   model = pickle.load(m_f)

with open(os.path.join(model_path, labels_file_name), "r") as l_f:
                labels_dict_list = json.load(l_f)

labels_dict = {d["label"]: d["answer_id"] for d in labels_dict_list}
transformer_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

embedder = Embedding(transformer_model, transformer_func)

tokenizer = TextsTokenizer()
text = "ORB 253 при попытки выгрузить акт приема передачи выходит ошибка (скрин во во=ложении) прошу исправить"
# text = "Когда получу документы по уже законченным курсам а то пропустила сроки сдачи экзаменов"

lm_text = " ".join(tokenizer([text])[0])
input_vector = embedder(text)
predict_labels = model.predict_proba([input_vector])

print(predict_labels)