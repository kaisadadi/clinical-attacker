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
record_type = {}
for piece in df:
    #1->SUBJECT_ID 2->HADM_ID
    print("Now is %d" %(cnt + 1))
    cnt += 1
    for idx in range(len(piece)):
        CATEGORY = piece.iloc[idx, 6]
        if CATEGORY not in record_type.keys():
            record_type[CATEGORY] = 0
        record_type[CATEGORY] += 1
        if CATEGORY != "Discharge summary":
            continue
        SUBJECT_ID = piece.iloc[idx, 1]
        HADM_ID = piece.iloc[idx, 2]
        raw_text = piece.iloc[idx, -1].lower()
        #locate history of present illness
        pos1 = raw_text.find("history of present illness:")
        if pos1 == -1:
            continue
        pos1 +=  + len("history of present illness:")
        pattern = re.compile(r"\n\n+[^\n]+?:")
        matched_pattern = re.findall(pattern, raw_text[pos1:])
        try:
            pos2 = raw_text[pos1:].find(matched_pattern[0])
        except:
            print("we died")
            continue
        if pos2 == -1:
            gg
        TEXT = raw_text[pos1 : pos2 + pos1]
        # locate history of present illness
        pos1 = raw_text.find("chief complaint:")
        if pos1 != -1:
            pos1 += len("chief complaint:")
            pattern = re.compile(r"\n\n[^\n]+?:")
            matched_pattern = re.findall(pattern, raw_text[pos1:])[0]
            pos2 = raw_text[pos1:].find(matched_pattern)
            if pos2 != -1:
                TEXT = raw_text[pos1:pos1 + pos2] + " " + TEXT
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
print(record_type)

