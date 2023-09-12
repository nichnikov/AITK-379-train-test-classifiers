import os
import pandas as pd
import fasttext
from random import shuffle
from texts_processing import TextsTokenizer
from config import PATH


# подготовка данных для обучения
tokenizer = TextsTokenizer()
dataset_df = pd.read_csv(os.path.join(PATH, "data", "test_dataset.csv"), sep="\t")

test_dataset = list(dataset_df.itertuples(index=False))
print(len(test_dataset))

test_data = []
for x in test_dataset:
    test_data.append(("__label__" + "_".join(str(x.label).split()), " ".join(tokenizer([str(x.text)])[0])))


        
model = fasttext.load_model(os.path.join(PATH, "models", "techsupport.model"))

true = 0
false = 0
for lb, text in test_data[:100]:
    pred = model.predict(text, k=1)
    if lb == pred[0][0]:
        true += 1
    else:
        false += 1
    print(lb, pred[0][0], pred[1][0])

print("true:", true, "false:", false)