import os
import json

data_dir = r"newutil/mt_snomed.txt"

CUI_dict = {}

def pre_examine():
    cnt = 0
    f = open(data_dir, "r", encoding="utf-8")
    for line in f.readlines():
        line = line.strip().split("##")
        CUI = line[3]
        title = line[1]
        if CUI not in CUI_dict.keys():
            CUI_dict[CUI] = []
        CUI_dict[CUI].append(title)

    key_num = 0
    sum_hit = 0

    for key in CUI_dict.keys():
        key_num += 1
        sum_hit += len(CUI_dict[key])

    print(key_num)
    print(float(sum_hit / key_num))

def gen_CUI_dict():
    #get CUI dict
    f = open(data_dir, "r", encoding="utf-8")
    for line in f.readlines():
        line = line.strip().split("##")
        CUI = line[3]
        title = line[1]
        if "退休的" in title:
            continue
        #去除左右括号
        if title.find("(") != -1:
            title = title[:title.find("(")]
        if title.find("（") != -1:
            title = title[:title.find("（")]
        if CUI not in CUI_dict.keys():
            CUI_dict[CUI] = set()
        CUI_dict[CUI].add(title)
    for key in CUI_dict.keys():
        CUI_dict[key] = list(CUI_dict[key])
    json.dump(CUI_dict, open("/data2/wk/data/pot/dict/concept.json", "w"))

if __name__ == "__main__":
    gen_CUI_dict()