import os
import pandas as pd
from transformers import (BertTokenizer,
                          BertModelWithHeads,
                          )
from classifier_tfidf.classifier_tfidf import index, tknz, dct, tfidf
import torch


def predict(premise):
    # encoded = tokenizer(premise, hypothesis, return_tensors="pt")
    encoded = tokenizer(premise, return_tensors="pt")
    # if torch.cuda.is_available():
    #  encoded.to("cuda")
    logits = model(**encoded)[0]
    pred_class = torch.argmax(logits).item()
    sigm = max(torch.sigmoid(logits)[0]).item()
    return pred_class, sigm

model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)

adapter_name = "classifier_adapter"
model_name = "classifier_adapter"

# adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path = os.path.join(os.getcwd(), "data", "models", model_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)

queries_df = pd.read_csv(os.path.join("data", "supp_queries_for_test.csv"), sep="\t")

"query"
texts = ["Как получить документ подтверждающий обучение", "Не успеваю пройти тест, можно сдать тест позже?", "накладная", 
         "надоел ваш спам отпишите меня от рассылки", "надоел ваш спам", "отпишите меня от рассылки", "я закончил курс, когда можно будет сдать тест", 
         "теперь я чебуршка мне каждая дворняжка", "сколько стоит дом построить",
"за инвалид травматизм",
"акция приз расход рекламный усна",
"2022 в взнос для ита компания процент страховой",
"график должный кто план утверждать",
"в взнос ли нужно платить самозанимать страховой фсс",
"модернизация оборудование по проводка",
"билет жд как лицо на от подотчетный принимать расход"
"запрашивать как об уведомление усна", "мне нужен сертификат, где я его могу скачать?", 
"где посмотреть сертификат с вебинара, я прошел вебинар, а сертификата нет"]

test_dicts = queries_df.to_dict(orient="records")
print(test_dicts[:10])

for text in texts:
    test_tokens = tknz([text])
    test_corpus = dct.doc2bow(test_tokens[0])
    test_vector = tfidf[test_corpus]
    sims = index[test_vector]

    cls, sgm = predict(text)
    print(text, "class:", cls, "score:", sgm, "tfidf scores:", sorted([(num, scr) for num, scr in enumerate(list(sims))], key=lambda x: x[1], reverse=True))
    # print(text, cls, sgm)