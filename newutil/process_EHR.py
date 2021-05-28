import os
import json
import pandas as pd

data_dir = r"/home/wke18/Changgeng-Hospital/Data/table.csv"
ICD_mapping_file = "/home/wke18/Changgeng-Hospital/Data/ICD-9-raw.txt"  #153
ICD_mapping_file = "/home/wke18/Changgeng-Hospital/Data/ICD-9-raw-after.txt"  #147
output_dir = r"/data2/wk/data/pot/rawdata.json"

def process_data():
    #读原始文件
    origin_df = pd.read_csv(data_dir, dtype='unicode')
    df = origin_df[["现病史", "主诉", "辅助检查", "体格检查", "急诊诊断"]]
    df = df.dropna(axis=0)
    print("df pre size = %d" %len(df))

    #ICD映射词表
    f = open(ICD_mapping_file, "r", encoding="gbk")
    word2ICD = {}
    IDC2word = {}
    for line in f.readlines():
        line = line[:-1].strip().split(" ")
        if len(line) == 1:
            continue
        chinese, ICD = line[0], line[1]
        if chinese not in word2ICD.keys():
            word2ICD[chinese] = ICD
            IDC2word[ICD] = chinese
    for key in ['780.6', '558.9', '465.9', '789.00', '401.9', '462', '786.2', '486', '780.4', '414.00', '250.00', '784.1', '276.8', '599.0', '530.11', '787.03', '434.1', '272.4', '466.0', '463']:
        print(key, IDC2word[key])

    #拼接文本+诊断结果转换
    texts = []
    diseases = []
    for idx in range(len(df)):
        texts.append(df.iloc[idx, 0] + df.iloc[idx, 1] + df.iloc[idx, 2] + df.iloc[idx, 3])
        disease = []
        diagnosis = df.iloc[idx, 4].split(" ")
        for diag in diagnosis:
            #先去问号
            diag = diag.strip('?？')
            #再去括号,
            if "（" in diag:
                left = diag.find("（")
                diag = diag[:left]
            if "(" in diag:
                left = diag.find("(")
                diag = diag[:left]
            pos = max(diag.find("."), diag.find("、")) + 1
            if diag[pos:] in word2ICD.keys():
                disease.append(word2ICD[diag[pos:]])
        diseases.append(disease)


    #导出结果
    assert len(texts) == len(diseases)
    f = open(output_dir, "w")
    for idx in range(len(texts)):
        out_data = {"text": texts[idx], "label": diseases[idx]}
        print(json.dumps(out_data), file=f)


def check_data():
    f = open(output_dir, "r")
    cnt = 0
    for line in f.readlines():
        data = json.loads(line)
        cnt += 1
    print("cnt = %d" %cnt)


if __name__ == "__main__":
    #check_data()
    process_data()
    
