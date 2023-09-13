import os
import pandas as pd
from config import PATH
from knn_model import knn_model_creater
from embeddings import Embedding, transformer_func
from sentence_transformers import SentenceTransformer
from texts_processing import TextsTokenizer


train_df = pd.read_csv(os.path.join(PATH, "data", "train_dataset_lb2int.csv"), sep="\t")
print(train_df)

transformer_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
tknz = TextsTokenizer()
train_df.rename(columns={"label": "ID",	"text": "Cluster"}, inplace=True)

transformer_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embedder = Embedding(transformer_model, transformer_func)

knn_model_creater(sys_id=10, neighbors=5, is_lematize=True, input_df=train_df, 
                  tokenizer=tknz, pubs_list=[246], embedder=embedder)