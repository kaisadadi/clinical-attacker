import os
import json
#from nltk.tokenize import RegexpTokenizer
import nltk
import multiprocessing
import time
#import warnings

#warnings.filterwarnings('ignore')

data_dir = r"/data/wke18/data/ADV/Final/VALID_pre"
metamap_dir = r"/home/wke18/MetaMap/public_mm/bin/metamap"
#tokenizer = RegexpTokenizer(r'[a-zA-Z]+|[.,()''""\-\!\?]')
tokenizer = nltk.word_tokenize

raw_data = []
q = multiprocessing.Queue()
num_process = 20

def load_data():
    global raw_data, q
    cnt = 0
    file_list = os.listdir(data_dir)
    for file in file_list:
        if file[-4:] != "json":
            continue
        raw_data.append(file)
    for idx in range(0, len(raw_data)):
        q.put(idx)

def work():
    while True:
        try:
            nowidx = q.get(timeout=5)
            full_name = os.path.join(data_dir, raw_data[nowidx])
            data = json.load(open(full_name, "r"))
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
                TEXT = tokenizer(raw_text)
                for idx, word in enumerate(TEXT):
                    if word.isupper():
                        segmented_pos.append(idx)
                piece["segented_pos"] = segmented_pos
                del piece["pos"]
                data["MetaMap"].append(piece)
            if '.json.json' in raw_data[nowidx]:
                print("happy")
                gg
            print(json.dumps(data), file=open(r"/data/wke18/data/ADV/Final/VALID_2/%s" %raw_data[nowidx], "w"))
            if os.path.exists("%s.json" %str(nowidx)):
                os.remove("%s.json" %str(nowidx))

        except multiprocessing.TimeoutError as e:
            print("I am timeout!!!")
            return


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
