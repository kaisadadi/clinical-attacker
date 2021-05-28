import os
import json
from nltk.tokenize import RegexpTokenizer
import multiprocessing
import time
import warnings

warnings.filterwarnings('ignore')

data_dir = r"/data/wke18/data/mimic/note_label.json"
metamap_dir = r"/home/wke18/MetaMap/public_mm/bin/metamap"
tokenizer = RegexpTokenizer(r'[a-zA-Z]+|[.,()''""\-\!\?]')

raw_data = []
q = multiprocessing.Queue()
num_process = 15

def load_data():
    global raw_data, q
    cnt = 0
    with open(data_dir, "r", encoding="utf-8") as f:
        for file in f:
            data = json.loads(file)
            raw_data.append(data)
    for idx in range(0, len(raw_data)):
        q.put(idx)


def work():
    while True:
        try:
            nowidx = q.get(timeout=5)
            data = raw_data[nowidx]
            data["TEXT"] = data["TEXT"].replace("\"", "").replace("\'", "")
            print(json.dumps(data), file=open(r"/data/wke18/data/ADV/data/%s.json" %str(nowidx), "w"))

        except multiprocessing.TimeoutError as e:
            return


def check_output():
    file_dir = r"/data/wke18/data/ADV_EHR/data"
    file_list = os.listdir(file_dir)
    show_up = []
    missed = []
    for file in file_list:
        num = int(file[:file.find(".")])
        show_up.append(num)
    for idx in range(100):
        if idx not in show_up:
            missed.append(idx)
    print(missed)


if __name__ == "__main__":
    #check_output()
    load_data()
    #for idx in [20, 31, 64, 83]:
    #    single_work(process_idx=idx)
        #break

    process_list = []
    for a in range(0, num_process):
        process_list.append(multiprocessing.Process(target=work))
    for a in range(0, num_process):
        process_list[a].start()

    while q.qsize() != 0:
        print("%d/%d" % (len(raw_data) - q.qsize(), len(raw_data)), end='\r')
        time.sleep(5)
