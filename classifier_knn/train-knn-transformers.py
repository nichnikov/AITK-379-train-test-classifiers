import os
import pandas as pd
from config import PATH
from knn_model_transformers import knn_model_creater
from texts_processing import TextsTokenizer
from transformers import AutoTokenizer, AutoModel

transformer_tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
transformer_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

train_df = pd.read_csv(os.path.join(PATH, "data", "train_dataset_lb2int.csv"), sep="\t")
print(train_df)

tknz = TextsTokenizer()
train_df.rename(columns={"label": "ID",	"text": "Cluster"}, inplace=True)

knn_model_creater(sys_id=10, neighbors=10, is_lematize=True, input_df=train_df, 
                  tokenizer=tknz, pubs_list=[246], transformer_model=transformer_model, transformer_tokenizer=transformer_tokenizer)