import os
import json
import random
import shutil

k = 10

#original_data_path = "/data/wke18/data/ADV/Final/data_10"
#train_data_path = "/data/wke18/data/ADV/Final/TRAIN"
#valid_data_path = "/data/wke18/data/ADV/Final/VALID_pre"
original_data_path = "/media/jdcloud/Train"
train_data_path = "/media/jdcloud/s1/TRAIN"
valid_data_path = "/media/jdcloud/s1/VALID"
patient_record = {}
all_num = 0
selected = set()

def get_patient_record():
    global all_num, patient_record
    file_list = os.listdir(original_data_path)
    all_num = len(file_list)
    for file in file_list:
        data = json.load(open(os.path.join(original_data_path, file), "r"))
        if data["SUBJECT_ID"] not in patient_record.keys():
            patient_record[data["SUBJECT_ID"]] = []
        patient_record[data["SUBJECT_ID"]].append(file)

def get_data_type(ratio = 0.9):
    global all_num, selected, patient_record
    selected_num = 0
    Endflag = 0
    while Endflag == 0:
        for key in patient_record.keys():
            rand_num = random.randint(0, 1)
            if rand_num == 1:
                for item in patient_record[key]:
                    if item not in selected:
                        selected_num += 1
                        selected.add(item)
                if selected_num > ratio * all_num:
                    Endflag = 1
                    break

def split_data():
    file_list = os.listdir(original_data_path)
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(valid_data_path):
        os.makedirs(valid_data_path)
    for file in file_list:
        if file in selected:
            shutil.copyfile(os.path.join(original_data_path, file), os.path.join(train_data_path, file))
        else:
            shutil.copyfile(os.path.join(original_data_path, file), os.path.join(valid_data_path, file))


if __name__ == "__main__":
    get_patient_record()
    get_data_type(ratio = 0.9)
    split_data()



