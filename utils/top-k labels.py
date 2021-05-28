import json
import os

original_data_path = "/data/wke18/data/ADV/Final/data"
label_calc = {}

def read_and_calc():
    global label_calc
    file_list = os.listdir(original_data_path)
    for file in file_list:
        if file[-5:] != ".json":
            continue
        data = json.load(open(os.path.join(original_data_path, file), "r"))
        for label in data["LABEL"]:
            if label not in label_calc.keys():
                label_calc[label] = 0
            label_calc[label] += 1


def calc_coverage(k = 50):
    global label_calc, sorted_label
    sorted_label = sorted(label_calc.items(), key=lambda x: x[1], reverse=True)
    label_list = []
    cover_num = 0
    disease2id = {}
    for idx in range(k):
        label_list.append(sorted_label[idx][0])
        disease2id[sorted_label[idx][0]] = idx
    json.dump(disease2id, open("/data/wke18/data/ADV/Final/disease2id_%s.json" %str(k), "w"))
    file_list = os.listdir(original_data_path)
    all_num = len(file_list)
    for file in file_list:
        data = json.load(open(os.path.join(original_data_path, file), "r"))
        in_flag = 0
        for label in data["LABEL"]:
            if label in label_list:
                in_flag = 1
                break
        if in_flag == 1:
            cover_num += 1
    print("ratio = ", cover_num / all_num)

def top_k_label_convert(k = 50):
    global sorted_label
    output_data_dir = "/data/wke18/data/ADV/Final/data_%s" %str(k)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    label_list = []
    label_2_num = {}
    for idx in range(k):
        label_2_num[sorted_label[idx][0]] = idx
        label_list.append(sorted_label[idx][0])
    file_list = os.listdir(original_data_path)
    for file in file_list:
        data = json.load(open(os.path.join(original_data_path, file), "r"))
        in_flag = 0
        filtered_label = []
        for label in data["LABEL"]:
            if label in label_list:
                in_flag = 1
                filtered_label.append(label_2_num[label])
        if in_flag == 1:
            data["LABEL"] = filtered_label
            json.dump(data, open(os.path.join(output_data_dir, file), "w"))


if __name__ == "__main__":
    read_and_calc()
    calc_coverage(k=10)
    top_k_label_convert(k=10)
    #result  top 50 90.5% ; top 100 95.77%
    #result  top 50 93.47%
    #v3 result: top 50 90.49% top 20 86.2% top 10 79.6%
    #v2 result: top 10 77%