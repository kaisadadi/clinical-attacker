import json
import pandas as pd

label_space = {}
cnt_label = {}
all_label_cnt = 0

def load_label_space():
    global label_space, cnt_label, all_label_cnt
    csv_data = pd.read_csv(r"/data/wke18/data/mimic/DIAGNOSES_ICD.csv")
    for idx in range(len(csv_data)):
        #0 ROW_ID, 1 SUBJECT_ID, 2 HADM_ID, 4_ICD9_CODE
        SUBJECT_ID = str(csv_data.iloc[idx, 1])
        HADM_ID = str(csv_data.iloc[idx, 2])
        ICD9_CODE = str(csv_data.iloc[idx, 4])
        if SUBJECT_ID not in label_space.keys():
            label_space[SUBJECT_ID] = {}
        if HADM_ID not in label_space[SUBJECT_ID].keys():
            label_space[SUBJECT_ID][HADM_ID] = []
        if ICD9_CODE not in label_space[SUBJECT_ID][HADM_ID]:
            label_space[SUBJECT_ID][HADM_ID].append(ICD9_CODE)
        #cnt label
        if ICD9_CODE not in cnt_label.keys():
            cnt_label[ICD9_CODE] = 0
        cnt_label[ICD9_CODE] += 1
        all_label_cnt += 1
    sorted_label = sorted(cnt_label.items(), key=lambda x:x[1], reverse=True)
    sum = 0
    for idx in range(100):
        sum += sorted_label[idx][1]
    print(float(sum / all_label_cnt))

def label_data():
    global label_space
    out_file = open(r"/data/wke18/data/mimic/note_label.json", "w")
    with open(r"/data/wke18/data/mimic/note.json", "r", encoding="utf-8") as file:
        cnt = 0
        cnt_all = 0
        for f in file:
            cnt_all += 1
            data = json.loads(f)
            SUBJECT_ID = data["SUBJECT_ID"]
            HADM_ID = data["HADM_ID"]
            if SUBJECT_ID in label_space.keys():
                if HADM_ID in label_space[SUBJECT_ID].keys():
                    out_data = {"TEXT": data["TEXT"],
                                "LABEL": label_space[SUBJECT_ID][HADM_ID],
                                "SUBJECT_ID": SUBJECT_ID,
                                "HADM_ID": HADM_ID}
                    cnt += 1
                    print(json.dumps(out_data), file=out_file)
    print(cnt)
    print(cnt_all)
    out_file.close()


if __name__ == "__main__":
    load_label_space()
    label_data()