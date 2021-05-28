import os
import json
import random

input_data_dir = r"/data2/wk/data/pot/label_processed.json"
output_train_path = r"/data2/wk/data/pot/train_data"
output_valid_path = r"/data2/wk/data/pot/valid_data"

ratio = 0.8

def split_data():
    if os.path.exists(output_train_path):
        os.system("rm -rf %s" %output_train_path)
    if os.path.exists(output_valid_path):
        os.system("rm -rf %s" %output_valid_path)
    os.mkdir(output_train_path)
    os.mkdir(output_valid_path)
    f = open(input_data_dir, "r")
    data = []
    for line in f.readlines():
        data.append(json.loads(line))
    random.shuffle(data)

    train_data_size = int(len(data) * ratio)
    for idx in range(train_data_size):
        json.dump(data[idx], open(os.path.join(output_train_path, "%d.json" %idx), "w"))
    for idx in range(train_data_size, len(data)):
        json.dump(data[idx], open(os.path.join(output_valid_path, "%d.json" %idx), "w"))


if __name__ == "__main__":
    split_data()



