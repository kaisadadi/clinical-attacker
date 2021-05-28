import pandas as pd
import json
import csv
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer, word_tokenize
import re


data_dir = r"/data/wke18/data/mimic/NOTEEVENTS.csv"
out_dir = r"/data/wke18/data/mimic/note.json"

tokenizer = RegexpTokenizer(r'[a-zA-Z]+|[.,()''""\-\!\?]')
df = pd.read_csv(data_dir, chunksize=1000)

f = open(out_dir, 'w', encoding='utf-8')
cnt = 0
for piece in df:
    #1->SUBJECT_ID 2->HADM_ID
    print("Now is %d" %(cnt + 1))
    cnt += 1
    for idx in range(len(piece)):
        CATEGORY = piece.iloc[idx, 6]
        if CATEGORY != "Discharge summary":
            continue
        SUBJECT_ID = piece.iloc[idx, 1]
        HADM_ID = piece.iloc[idx, 2]
        TEXT = piece.iloc[idx, -1].lower()
        #去除方括号
        pattern = re.compile(r"\[\*\*.+?\*\*\]")
        matched_pattern = re.findall(pattern, TEXT)
        for item in matched_pattern:
            TEXT = TEXT.replace(item, "")
        #TEXT = tokenizer.tokenize(TEXT)
        TEXT = TEXT.replace("\n", " ")
        data = {"SUBJECT_ID": str(SUBJECT_ID), "HADM_ID": str(HADM_ID), "TEXT": TEXT}
        print(json.dumps(data),file=f)
        #print(data)
    #break

