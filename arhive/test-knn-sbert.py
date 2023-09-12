import os
import json
import pickle
from classifier_knn.embeddings import (Embedding,
                                   transformer_func)
import pandas as pd
from src.config import PROJECT_ROOT_DIR
from sentence_transformers import SentenceTransformer
from classifier_tfidf.classifier_tfidf import index, tknz, dct, tfidf

model_prefix = "20230602_1013_SysId10"
model_path = os.path.join(PROJECT_ROOT_DIR, "data", "models", model_prefix)
model_name = model_prefix + "_knn.model"
labels_file_name = model_prefix + "_labels.json"

with open(os.path.join(model_path, model_name), "rb") as m_f:
   model = pickle.load(m_f)

with open(os.path.join(model_path, labels_file_name), "r") as l_f:
                labels_dict = json.load(l_f)

transformer_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

embedder = Embedding(transformer_model, transformer_func)

texts = texts = ["Как получить документ подтверждающий обучение", "Не успеваю пройти тест, можно сдать тест позже?", "накладная", 
         "надоел ваш спам отпишите меня от рассылки", "надоел ваш спам", "отпишите меня от рассылки", "я закончил курс, когда можно будет сдать тест", 
         "теперь я чебуршка мне каждая дворняжка", "сколько стоит дом построить",
"за инвалид травматизм",
"акция приз расход рекламный усна",
"2022 в взнос для ита компания процент страховой",
"график должный кто план утверждать",
"в взнос ли нужно платить самозанимать страховой фсс",
"модернизация оборудование по проводка",
"билет жд как лицо на от подотчетный принимать расход"
"запрашивать как об уведомление усна"]

labels = pd.DataFrame(labels_dict)
for text in texts:
   test_tokens = tknz([text])
   test_corpus = dct.doc2bow(test_tokens[0])
   test_vector = tfidf[test_corpus]
   sims = index[test_vector]

   input_vector = embedder(text)
   predict_labels = model.predict_proba([input_vector])
   ans_scs = sorted([x for x in zip(labels["answer_id"], predict_labels[0]) if x[1] > 0.5],
                                 key=lambda x: x[1], reverse=True)
   ans_scs_dct = {x[0]: x[1] for x in ans_scs}
   # print(text, predict_labels[0])
   print(text, ans_scs_dct, "tfidf scores:", sorted([(num, scr) for num, scr in enumerate(list(sims))], key=lambda x: x[1], reverse=True))