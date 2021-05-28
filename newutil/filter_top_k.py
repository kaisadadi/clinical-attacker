import os
import shutil
import json

train_dir = r"/data2/wk/data/pot/train_data"
valid_dir = r"/data2/wk/data/pot/test_data"
filtered_train_dir = r"/data2/wk/data/pot/train_data_20"
filtered_valid_dir = r"/data2/wk/data/pot/test_data_20"

def filter_data(k = 20, input_dir = None, output_dir = None):
    file_list = os.listdir(input_dir)
    for file in file_list:
        file = os.path.join(input_dir, file)
        data = json.load(open(file, "r"))
        labels, filterd_labels = data["label"], []
        for label in labels:
            if label < k:
                filterd_labels.append(label)
        if len(filterd_labels) <= 0:
            continue
        data["label"] = filterd_labels
        json.dump(data, open(file.replace(input_dir, output_dir), "w"))


if __name__ == "__main__":
    if os.path.exists(filtered_train_dir):
        os.system("rm -rf %s" %filtered_train_dir)
    if os.path.exists(filtered_valid_dir):
        os.system("rm -rf %s" %filtered_valid_dir)
    os.mkdir(filtered_train_dir)
    os.mkdir(filtered_valid_dir)
    filter_data(20, train_dir, filtered_train_dir)
    filter_data(20, valid_dir, filtered_valid_dir)