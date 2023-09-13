import os
from datasets import Dataset, DatasetDict
import pandas as pd
from config import PATH

queries_df = pd.read_csv(os.path.join(PATH, "data", "train_dataset_lb2int.csv"), sep="\t")
print(queries_df)

dtset = Dataset.from_pandas(queries_df)
train_data = DatasetDict({"train": dtset})
print(train_data)
train_data.save_to_disk(os.path.join(PATH, "data", "train.data"))