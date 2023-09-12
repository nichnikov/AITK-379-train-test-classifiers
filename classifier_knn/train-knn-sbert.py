import os
from knn_model import knn_model_creater
import pandas as pd
from texts_processing import TextsTokenizer
from config import PATH

train_df = pd.read_csv(os.path.join(PATH, "data", "trained_queries.csv"), sep="\t")
print(train_df)

tknz = TextsTokenizer()
train_df.rename(columns={"label": "ID",	"text": "Cluster"}, inplace=True)

knn_model_creater(sys_id=10, neighbors=10, is_lematize=True, input_df=train_df,
                      tokenizer=tknz, pubs_list=[246])