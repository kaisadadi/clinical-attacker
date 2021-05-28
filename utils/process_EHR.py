import os
import json
from nltk.tokenize import RegexpTokenizer
import multiprocessing
import time
#import warnings

#warnings.filterwarnings('ignore')

data_dir = r"/data/wke18/data/mimic/note_label.json"
metamap_dir = r"/home/wke18/MetaMap/public_mm/bin/metamap"
tokenizer = RegexpTokenizer(r'[a-zA-Z]+|[.,()''""\-\!\?]')

raw_data = []
q = multiprocessing.Queue()
num_process = 20

def load_data():
    global raw_data, q
    cnt = 0
    with open(data_dir, "r", encoding="utf-8") as f:
        for file in f:
            data = json.loads(file)
            raw_data.append(data)
    for idx in range(0, 5000):
        q.put(idx)

def work():
    while True:
        try:
            nowidx = q.get(timeout=5)
            data = raw_data[nowidx]
            data["TEXT"] = data["TEXT"].replace("\"", "").replace("\'", "")
            # metamap processing
            #return_value = os.system("echo \"{}\" ""| {} -I -t -s --silent --JSONf 2 -J \
            #            \"clnd\",\"sosy\",\"antb\",\"anab\",\"cgab\",\"dsyn\",\"inpo\",\"mobd\" \
            #            >\"{}.json\"".format(data["TEXT"], metamap_dir, str(nowidx)))
            return_value = os.system("echo \"{}\" ""| {} -I -t -s --silent --JSONf 2 -J \
                                    \"clnd\",\"sosy\",\"antb\",\"anab\",\"cgab\",\"dsyn\",\"inpo\",\"mobd\" \
                                    >\"{}.json\" 2> log".format(data["TEXT"], metamap_dir, str(nowidx)))
            if return_value != 0:
                if os.path.exists("%s.json" % str(nowidx)):
                    os.remove("%s.json" % str(nowidx))
                continue
            # remove first lines
            #return_value = os.system("sed -i \'1d\' %s.json" %str(nowidx))
            return_value = os.system("sed -i \'1d\' %s.json 2> removelog > gglog" % str(nowidx))
            if return_value != 0:
                if os.path.exists("%s.json" % str(nowidx)):
                    os.remove("%s.json" % str(nowidx))
                continue
            # load file from output
            try:
                out_json = json.load(open("%s.json" %str(nowidx), "r"))
            except:
                if os.path.exists("%s.json" % str(nowidx)):
                    os.remove("%s.json" % str(nowidx))
                continue
            # get_needed_answer
            out_mapping = []
            used_CUI = []
            assert len(out_json["AllDocuments"]) == 1
            for item in out_json["AllDocuments"][0]["Document"]["Utterances"]:
                phrases = item["Phrases"]
                for phs in phrases:
                    if phs["Mappings"] != []:
                        for mapping in phs["Mappings"]:
                            for m_can in mapping["MappingCandidates"]:
                                CUI = m_can["CandidateCUI"]
                                if CUI in used_CUI:
                                    break
                                else:
                                    used_CUI.append(CUI)
                                matched = m_can["CandidateMatched"]
                                pos = m_can["ConceptPIs"]
                                type = m_can["SemTypes"]
                                out_mapping.append({"CUI": CUI,
                                            "pos": pos,
                                            "matched": matched,
                                            "type": type})
            data["MetaMap"] = []
            for piece in out_mapping:
                raw_text = data["TEXT"]
                segmented_pos = []
                for pos in piece["pos"]:
                    raw_text = raw_text[:int(pos["StartPos"])] + raw_text[int(pos["StartPos"]): int(pos["StartPos"]) + int(
                        pos["Length"])].upper() \
                            + raw_text[int(pos["StartPos"]) + int(pos["Length"]):]
                TEXT = tokenizer.tokenize(raw_text)
                for idx, word in enumerate(TEXT):
                    if word.isupper():
                        segmented_pos.append(idx)
                piece["segented_pos"] = segmented_pos
                del piece["pos"]
                data["MetaMap"].append(piece)
            print(json.dumps(data), file=open(r"/data/wke18/data/ADV/data/%s.json" %str(nowidx), "w"))
            if os.path.exists("%s.json" %str(nowidx)):
                os.remove("%s.json" %str(nowidx))

        except multiprocessing.TimeoutError as e:
            print("I am timeout!!!")
            return

def single_work(process_idx):
    nowidx = process_idx
    data = raw_data[nowidx]
    data["TEXT"] = data["TEXT"].replace("\"", "").replace("\'", "")
    # metamap processing
    return_value = os.system("echo \"{}\" ""| {} -I -t -s --silent --JSONf 2 -J \
                            \"clnd\",\"sosy\",\"antb\",\"anab\",\"cgab\",\"dsyn\",\"inpo\",\"mobd\" \
                            >\"{}.json\" 2> \"\dev\\null\"".format(data["TEXT"], metamap_dir, str(nowidx)))
    if return_value != 0:
        return
    # remove first lines
    return_value = os.system("sed -i \'1d\' %s.json" % str(nowidx))
    if return_value != 0:
        return
    # load file from output
    try:
        out_json = json.load(open("%s.json" % str(nowidx), "r"))
    except:
        return
    # get_needed_answer
    out_mapping = []
    used_CUI = []
    assert len(out_json["AllDocuments"]) == 1
    for item in out_json["AllDocuments"][0]["Document"]["Utterances"]:
        phrases = item["Phrases"]
        for phs in phrases:
            if phs["Mappings"] != []:
                for mapping in phs["Mappings"]:
                    for m_can in mapping["MappingCandidates"]:
                        CUI = m_can["CandidateCUI"]
                        if CUI in used_CUI:
                            break
                        else:
                            used_CUI.append(CUI)
                        matched = m_can["CandidateMatched"]
                        pos = m_can["ConceptPIs"]
                        type = m_can["SemTypes"]
                        out_mapping.append({"CUI": CUI,
                                            "pos": pos,
                                            "matched": matched,
                                            "type": type})
    data["MetaMap"] = []
    for piece in out_mapping:
        raw_text = data["TEXT"]
        segmented_pos = []
        for pos in piece["pos"]:
            raw_text = raw_text[:int(pos["StartPos"])] + raw_text[int(pos["StartPos"]): int(pos["StartPos"]) + int(
                        pos["Length"])].upper() \
                                   + raw_text[int(pos["StartPos"]) + int(pos["Length"]):]
        TEXT = tokenizer.tokenize(raw_text)
        for idx, word in enumerate(TEXT):
            if word.isupper():
                segmented_pos.append(idx)
        piece["segented_pos"] = segmented_pos
        del piece["pos"]
        data["MetaMap"].append(piece)
    print(json.dumps(data), file=open(r"/data/wke18/data/ADV_EHR/data/%s.json" % str(nowidx), "w"))
    #if os.path.exists("%s.json" % str(nowidx)):
    #    os.remove("%s.json" % str(nowidx))

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
        print("%d/%d" % (5000 - q.qsize(), 5000), end='\r')
        time.sleep(5)
