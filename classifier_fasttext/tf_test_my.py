import os
import pandas as pd
import fasttext
from random import shuffle
from texts_processing import TextsTokenizer
from sklearn.metrics import confusion_matrix


# подготовка данных для обучения
tokenizer = TextsTokenizer()
dataset_df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_dataset_lb2int.csv"), sep="\t")

test_dataset = list(dataset_df.itertuples(index=False))
print(len(test_dataset))

test_data = []
for x in test_dataset:
    test_data.append(("__label__" + "_".join(str(x.label_str).split()), " ".join(tokenizer([str(x.text)])[0])))


        
model = fasttext.load_model(os.path.join(os.getcwd(), "models", "techsupport.model"))

print(model.get_labels())
unique_labels = {lb: num for num, lb in enumerate(model.get_labels())}
print(unique_labels, len(unique_labels))


results = []
true_pred = 0
not_other_pred = 0
k = 1
for true_label, text in test_data:
    predict_labels = model.predict(text, k=1)
    if predict_labels[0][0] == true_label:
        true_pred += 1
    if predict_labels[0][0] != "__label__Другое":
        not_other_pred += 1
    results.append({"text": text,
                    "true_label": true_label, 
                    "predict_label": predict_labels[0][0]})
    print(k, "/", len(test_data), "true:", true_label, "pred:", predict_labels[0][0])
    k += 1

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(os.getcwd(), "data", "test_results.tsv"), sep="\t", index=False)
print("true_pred:", true_pred, "precision:", true_pred/not_other_pred)
print("not_other_pred:", not_other_pred, "recall:", not_other_pred/len(test_data))