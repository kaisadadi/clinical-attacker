import json

data_dir = r"/data/wke18/data/ADV_EHR/data"

idx = 2

data = json.load(open(data_dir + "/" + "%s.json" %str(idx), "r"))

print(data)