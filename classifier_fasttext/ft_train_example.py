"""http://ethen8181.github.io/machine-learning/deep_learning/multi_label/fasttext.html"""
"""https://thinkinfi.com/fasttext-for-text-classification-python/"""

import os
import fasttext
import random
from config import PATH

def train_test_split_file(input_path: str,
                          output_path_train: str,
                          output_path_test: str,
                          test_size: float,
                          random_state: int=1234,
                          encoding: str='utf-8',
                          verbose: bool=True):
    random.seed(random_state)

    # we record the number of data in the training and test
    count_train = 0
    count_test = 0
    train_range = 1 - test_size

    with open(input_path, encoding=encoding) as f_in, \
         open(output_path_train, 'w', encoding=encoding) as f_train, \
         open(output_path_test, 'w', encoding=encoding) as f_test:

        for line in f_in:
            random_num = random.random()
            if random_num < train_range:
                f_train.write(line)
                count_train += 1
            else:
                f_test.write(line)
                count_test += 1

    if verbose:
        print('train size: ', count_train)
        print('test size: ', count_test)


def prepend_file_name(path: str, name: str) -> str:
    """
    e.g. data/cooking.stackexchange.txt
    prepend 'train' to the base file name
    data/train_cooking.stackexchange.txt
    """
    directory = os.path.dirname(path)
    file_name = os.path.basename(path)
    return os.path.join(directory, name + '_' + file_name)

if __name__ == "__main__":
    data_dir = 'data'
    test_size = 0.2
    input_path = os.path.join(PATH, data_dir, 'cooking.stackexchange.txt')
    input_path_train = prepend_file_name(input_path, 'train')
    input_path_test = prepend_file_name(input_path, 'test')
    random_state = 1234
    encoding = 'utf-8'
    """
    train_test_split_file(input_path, input_path_train, input_path_test,
                        test_size, random_state, encoding)
    print('train path: ', input_path_train)
    print('test path: ', input_path_test)

    fasttext_params = {
    'input': input_path_train,
    'lr': 0.1,
    'lrUpdateRate': 1000,
    'thread': 8,
    'epoch': 10,
    'wordNgrams': 1,
    'dim': 100,
    'loss': 'ova'
    }
    
    model = fasttext.train_supervised(**fasttext_params)
    model.save_model(os.path.join(PATH, "models", "cooking.model"))

    print('vocab size: ', len(model.words))
    print('label size: ', len(model.labels))
    print('example vocab: ', model.words[:5])
    print('example label: ', model.labels[:5])"""

    model = fasttext.load_model(os.path.join(PATH, "models", "cooking.model"))
    # text = 'How much does potato starch affect a cheese sauce recipe?'
    text = "Will dried buttermilk react with other ingredients in dry mix with oil, stored in fridge?"
    # text = "Safe Chicken Liver?"
    pred = model.predict(text, k=2)
    print(pred)