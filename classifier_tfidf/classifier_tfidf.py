import os
import pandas as pd
from gensim.similarities import MatrixSimilarity
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from src.texts_processing import TextsTokenizer
from itertools import chain
import itertools
import operator

def group_by_lbs(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
       yield key, list(item[1] for item in subiter)

tknz = TextsTokenizer()
train_df = pd.read_csv(os.path.join("data", "trained_queries.csv"), sep="\t")

groups_texts = list(zip(train_df["label"], train_df["text"]))
texts_by_groups = sorted(list(group_by_lbs(groups_texts)), key=lambda x: x[0])


texs = list(train_df["text"])
tokens = tknz(texs)
dct = Dictionary(tokens)

texts_by_groups_tokenized = [[x for x in chain(*tknz(txs))] for grp, txs in texts_by_groups]
corpus = [dct.doc2bow(item) for item in texts_by_groups_tokenized]

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

print(corpus_tfidf)
index = MatrixSimilarity(tfidf[corpus],  num_features=len(dct))

'''
test_texts = ["Как получить документ подтверждающий обучение", "Не успеваю пройти тест, можно сдать тест позже?", "накладная", 
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

for test_text in test_texts:
    test_tokens = tknz([test_text])

    test_corpus = dct.doc2bow(test_tokens[0])
    test_vector = tfidf[test_corpus]
    sims = index[test_vector]

    print(test_text, "max score:", max(sims), "group number:", list(sims).index(max(sims)))'''