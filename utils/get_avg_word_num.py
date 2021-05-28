import json
import os
from nltk.tokenize import RegexpTokenizer

file_dir = r"/data/wke18/data/ADV_EHR/DATA_50/TRAIN_50"
tokenizer = RegexpTokenizer(r'[a-zA-Z]+|[.,()''""\-\!\?]')
file_list = os.listdir(file_dir)

sum = 0

for file in file_list:
    full_file = os.path.join(file_dir, file)
    data = json.load(open(full_file, "r"))
    text = data['TEXT']
    text = tokenizer.tokenize(text)
    sum += len(text)

print(sum / len(file_list))