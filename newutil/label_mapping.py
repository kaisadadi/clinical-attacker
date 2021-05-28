import json
import random
import os

raw_data_dir = r"/data2/wk/data/pot/rawdata.json"
output_dir = r"/data2/wk/data/pot/label_processed.json"
ratio = 0.8
label_cnt = {}
top_k = {}

def gen_top_k(k = 50):
    f = open(raw_data_dir, "r")
    for line in f.readlines():
        data = json.loads(line)
        for label in data["label"]:
            if label not in label_cnt.keys():
                label_cnt[label] = 0
            label_cnt[label] += 1
    sorted_label = sorted(label_cnt.items(), key = lambda x: x[1], reverse = True)
    ICD_list = []
    for idx in range(k):
        top_k[sorted_label[idx][0]] = idx
        print(idx, sorted_label[idx][0])
        ICD_list.append(sorted_label[idx][0])
    print(ICD_list)
    

def gen_train_eval_data():
    f = open(raw_data_dir, "r")
    out_data = []
    for line in f.readlines():
        data = json.loads(line)
        digit_label = []
        for label in data["label"]:
            if label in top_k.keys():
                digit_label.append(top_k[label])
        if len(digit_label) > 0:
            out_data.append({"text": data["text"], "label": digit_label})
    f = open(output_dir, "w")
    for item in out_data:
        print(json.dumps(item), file=f)
    

if __name__ == "__main__":
    gen_top_k(k = 20)
    #gen_train_eval_data()