'''
Тестирование работы алгоритма на тестовых данных 
код взят отсюда: https://ethen8181.github.io/machine-learning/deep_learning/multi_label/fasttext.html
'''
import os
import pandas as pd
import fasttext
from random import shuffle
from texts_processing import TextsTokenizer
from sklearn.metrics import confusion_matrix
from config import data_path

def prepend_file_name(path: str, name: str) -> str:
    """
    e.g. data/cooking.stackexchange.txt
    prepend 'train' to the base file name
    data/train_cooking.stackexchange.txt
    """
    directory = os.path.dirname(path)
    file_name = os.path.basename(path)
    return os.path.join(directory, name + '_' + file_name)


def print_results(model, input_path, k):
    num_records, precision_at_k, recall_at_k = model.test(input_path, k)
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    print("records\t{}".format(num_records))
    print("Precision@{}\t{:.3f}".format(k, precision_at_k))
    print("Recall@{}\t{:.3f}".format(k, recall_at_k))
    print("F1@{}\t{:.3f}".format(k, f1_at_k))
    print()


if __name__  == "__main__":
    model_name = "corpsite.model"
    input_path = os.path.join(os.getcwd(), "data", 'train_data.txt')
    input_path_train = prepend_file_name(input_path, 'train')
    input_path_test = prepend_file_name(input_path, 'test')
    model = fasttext.load_model(os.path.join(os.getcwd(), "models", model_name))
    
    k = 1 # количество "лейблов" для входящего вопроса
    print('train metrics:')
    print_results(model, input_path_train, k)

    print('test metrics:')
    print_results(model, input_path_test, k)