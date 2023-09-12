import os
import pandas as pd
import fasttext
from random import shuffle
from texts_processing import TextsTokenizer
from sklearn.metrics import confusion_matrix
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

print(model.get_labels())
unique_labels = {lb: num for num, lb in enumerate(model.get_labels())}
print(unique_labels, len(unique_labels))

true = 0
false = 0
y_true = []
y_predict = []
for lb, text in test_data:
    pred = model.predict(text, k=1)
    if pred[0][0] not in unique_labels:
        unique_labels[pred[0][0]] = len(unique_labels)
        print(unique_labels)
    if lb  not in unique_labels:
        unique_labels[lb] = len(unique_labels)
        print(unique_labels)
    try:
        y_true.append(unique_labels[lb])
        y_predict.append(unique_labels[pred[0][0]])
    except:
        pass
        # print("can't add label:", lb, "or", pred[0][0])
    if lb == pred[0][0]:
        true += 1
    else:
        false += 1
    # print(lb, pred[0][0], pred[1][0])

print("true:", true, "false:", false)
cm = confusion_matrix(y_true, y_predict)
print(cm)