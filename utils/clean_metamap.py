import json
import os

data_dir = r"/data/wke18/data/ADV/Final/VALID_2"
output_data_dir = r"/data/wke18/data/ADV/Final/VALID_3"
file_list = os.listdir(data_dir)

for file in file_list:
    full_file = os.path.join(data_dir, file)
    data = json.load(open(full_file, "r"))
    Metamap = []
    for k1 in range(len(data["MetaMap"])):
        flag = 0
        try:
            k1_lb = min(data["MetaMap"][k1]["segented_pos"])
            k1_ub = max(data["MetaMap"][k1]["segented_pos"])
        except:
            continue
        for k2 in range(len(Metamap)):
            if k1 == k2:
                continue
            #判断是否相交
            try:
                k2_lb = min(Metamap[k2]["segented_pos"])
                k2_ub = max(Metamap[k2]["segented_pos"])
            except:
                flag = 1
                break
            if k2_lb > k1_ub or k1_lb > k2_ub:
                pass
            else:
                flag = 1
                break
        if flag == 0:
            Metamap.append(data["MetaMap"][k1])
    data["MetaMap"] = Metamap
    json.dump(data, open(os.path.join(output_data_dir, file), "w"))
