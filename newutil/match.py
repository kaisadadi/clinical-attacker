import os
import json
import numpy as np
import time
import numba as nb
import functools
import multiprocessing

#先load实体表到空间内
#dp, 按照病历分20组, 开20进程

entity_data_dir = r"newutil/mt_snomed.txt"
#note_data_dir = r"/data2/wk/data/pot/label_processed.json"
valid_data_path = r"/data2/wk/data/pot/valid_data"
processed_data_path = r"/data2/wk/data/pot/test_data"
num_process = 24    
max_note_len = 400
max_entity_len = 10
entity_dict = {}  #entity: [idx, CUI]

q = multiprocessing.Queue()

def load_entity():
    f = open(entity_data_dir, "r", encoding="utf-8")
    cnt = 0
    for line in f.readlines():
        line = line.strip().split("##")
        CUI = line[3]
        title = line[1]
        if title.find("(") != -1:
            title = title[:title.find("(")]
        if title.find("（") != -1:
            title = title[:title.find("（")]
        if title in entity_dict.keys():
            continue
        if len(title) < 2 or len(title) > 10:
            continue
        flag = 1
        for ch in title:
            if ch < u'\u4e00' or ch > u'\u9fff':
                flag = 0
                break
        if flag == 0:
            continue
        entity_dict[title] = [cnt, CUI]
        cnt += 1

@nb.jit(nopython=True) 
def pair_work(note, entity, tolerance, f):
    #单note+单entity匹配
    #编辑距离上界为tolerance(float，即完美匹配的百分比)
    len_note, len_entity = len(note), len(entity)
    f = f * 0
    score = {"equal": 3, "mismatch": -1, "indel": -1}
    full_score = len_entity * score["equal"]
    least_score = int(full_score * tolerance)
    for i in range(1, len_note + 1):
        for j in range(1, len_entity + 1):
            if note[i-1] == entity[j-1]:
                f[i][j] = max(f[i][j], f[i-1][j-1] + score["equal"])    
            else:
                f[i][j] = max(f[i][j], f[i-1][j-1] + score["mismatch"])
            f[i][j] = max(f[i][j], max(f[i][j-1],f[i-1][j]) + score["indel"])
            if (len_entity - j) * score["equal"] + f[i][j] < least_score:
                break
            j+=1
        i+=1
    max_grade, posx, posy = 0, 0, len_entity
    for i in range(1, len_note + 1):
        if f[i][len_entity] > max_grade:
            max_grade = f[i][len_entity]
            posx = i
    if max_grade < least_score:
        return 0, [-1, -1]    #代表无法按照要求匹配上
    #print(entity, max_grade, posx)
    right_border = posx
    while f[posx][posy] != 0:
        if note[posx-1] == entity[posy-1]:
            if f[posx-1][posy-1] + score["equal"] == f[posx][posy]:
                posx = posx - 1
                posy = posy - 1
                continue
        if f[posx-1][posy-1] + score["mismatch"] == f[posx][posy]:
            posx = posx - 1
            posy = posy - 1
            continue
        if f[posx][posy-1] + score["indel"] == f[posx][posy]:
            posy = posy - 1
            continue
        if f[posx-1][posy] + score["indel"] == f[posx][posy]:
            posx = posx - 1
            continue
    return max_grade, [posx, right_border]

def single_work():
    while True:
        try:
            file_path = q.get(timeout=5)
            data = json.load(open(os.path.join(valid_data_path, file_path), "r"))
            note = data["text"]
            if len(note) > max_note_len:
                note = note[:max_note_len]
            begin_time = time.time()
            cnt = 0
            ans_set = []
            dp = np.zeros([max_note_len + 1, max_entity_len + 1], np.int32)
            for key in entity_dict.keys():
                if len(key) < 5:
                    tolerance = 0.9   
                else:
                    tolerance = float((3 * len(key) - 4 - 0.5) / (3 * len(key)))
                if len(key) > max_entity_len:
                    key = key[:max_entity_len]
                cnt += 1
                grade, [left, right] = pair_work(note, key, tolerance, dp)
                if left != -1:
                    ans_set.append({"left": left, "right": right, "score": grade, "CUI": entity_dict[key][1], "name": key})
            #后处理
            sorted_ans = sorted(ans_set, key = lambda x: x["left"])
            out_res = {"entity": []}
            idx = 0
            while idx < len(sorted_ans):
                k = idx + 1
                max_right = sorted_ans[idx]["right"]
                max_score = sorted_ans[idx]["score"]
                max_pos = idx
                while k < len(sorted_ans) and sorted_ans[k]["left"] == sorted_ans[idx]["left"]:
                    if sorted_ans[k]["right"] > max_right:
                        max_right = sorted_ans[k]["right"]
                        max_pos = k
                        max_score = sorted_ans[k]["score"]
                    elif sorted_ans[k]["right"] == max_right:
                        if sorted_ans[k]["score"] > max_score:
                            max_score = sorted_ans[k]["score"]
                            max_pos = k
                    k += 1
                out_res["entity"].append({"left": sorted_ans[max_pos]["left"],
                                            "right": sorted_ans[max_pos]["right"],
                                            "name": sorted_ans[max_pos]["name"],
                                            "CUI": sorted_ans[max_pos]["CUI"]
                })
                idx = k
            
            out_res["note"] = data["text"]
            out_res["label"] = data["label"]
            json.dump(out_res, open(os.path.join(processed_data_path, file_path), "w"))
        except multiprocessing.TimeoutError as e:
            print("A process is timeout!!!")
            return

if __name__ == "__main__":
    load_entity()
    file_list = os.listdir(valid_data_path)
    for file in file_list:
        q.put(file)
    process_list = []
    for a in range(0, num_process):
        process_list.append(multiprocessing.Process(target=single_work))
    for a in range(0, num_process):
        process_list[a].start()

    while q.qsize() != 0:
        print("%d/%d" % (len(file_list) - q.qsize(), len(file_list)), end='\r')
        time.sleep(5)
    
    for a in range(0, num_process):
        process_list[a].join()