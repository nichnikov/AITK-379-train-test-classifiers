import os
import pandas as pd
from random import shuffle
from texts_processing import TextsTokenizer

# подготовка данных для обучения
tokenizer = TextsTokenizer()
dataset_df = pd.read_csv(os.path.join(os.getcwd(), "data", "train_dataset.csv"), sep="\t")
print(dataset_df)

train_dataset = list(dataset_df.itertuples(index=False))
print(len(train_dataset))

train_data = []
for x in train_dataset:
    train_data.append("__label__" + "_".join(str(x.label).split()) + " " + " ".join(tokenizer([str(x.text)])[0]))

shuffle(train_data)

with open(os.path.join(os.getcwd(), "data", "train_data.txt"), "w", encoding="utf-8") as f:
    for ln in train_data:
        f.write(str(ln) + "\n")