import requests
import json
from lxml.html import fromstring
import multiprocessing
import time

uri = "https://utslogin.nlm.nih.gov"
api_key = "7f9fc65e-ab9e-4adf-92c9-a67a3c81dacf"
auth_endpoint = "/cas/v1/api-key"


def get_ticket():
    params = {'apikey': api_key}
    h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
    r = requests.post(uri + auth_endpoint, data=params, headers=h)
    response = fromstring(r.text)
    tgt = response.xpath('//form/@action')[0]
    return tgt


def get_service_ticket(ticket):
    params = {'service': "http://umlsks.nlm.nih.gov"}
    h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
    r = requests.post(ticket, data=params, headers=h)
    st = r.text
    return st


def get_CUI_info(ticket, CUI):
    url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/"
    service_ticket = None
    while True:
        #print("happy")
        service_ticket = get_service_ticket(ticket)
        if len(service_ticket) > 5:
            break
    info = requests.get(url=url + CUI + "/atoms",
                        # headers={"content-type": 'application/x-www-form-urlencoded'},
                        params={"ticket": service_ticket, "language": 'ENG'})
    info.encoding = 'utf-8'
    try:
        items = json.loads(info.text)
        jsonData = items["result"]
        atom_set = set()
        for idx in range(len(jsonData)):
            atom = jsonData[idx]['name'].lower()
            if atom not in atom_set:
                atom_set.add(atom)
        atom_list = [item for item in atom_set]
        return atom_list
    except:
        return []

def sub_process_work(ticket):
    while True:
        try:
            CUI = q.get(timeout=5)
            atom_list = get_CUI_info(ticket, CUI)
            ans_q.put({CUI: atom_list})
        except multiprocessing.TimeoutError as e:
            return

def main_process_work(need_calc):
    print("main work start!")
    concept_list = {}
    while True:
        try:
            atom_dict = ans_q.get()
            for key in atom_dict.keys():
                if key not in concept_list.keys():
                    concept_list[key] = atom_dict[key]
            if len(concept_list) == need_calc:
                break
            else:
                print("%d/%d" % (len(concept_list), need_calc), end='\r')
        except multiprocessing.TimeoutError as e:
            break
    return concept_list

if __name__ == "__main__":
    #print(get_ticket())
    ticket = "https://utslogin.nlm.nih.gov/cas/v1/api-key/TGT-852234-x9QCrE0BEBBV4JveNIjwWPugOeCGWciW1eZ9kno9nG4pExAuKJ-cas"
    #get_CUI_info(ticket, "C0985346")

    q = multiprocessing.Queue()
    ans_q = multiprocessing.Queue()
    num_process = 20
    concept_list = json.load(open("/data/wke18/data/ADV/Final/concept_v3.json", "r"))
    cnt = 0
    need_calc = 0
    no_need = 0
    for key in concept_list.keys():
        if concept_list[key] == []:
            need_calc += 1
            q.put(key)
            if need_calc >= 2000:
                break
        else:
            no_need += 1
    print("need_calc:", need_calc)
    print("no_need", no_need)
    gg
    #start multi-process
    process_list = []
    for a in range(0, num_process):
        print("prepare #%d" %a)
        process_list.append(multiprocessing.Process(target=sub_process_work, args=(ticket, )))
    for a in range(0, num_process):
        print("start #%d" %a)
        process_list[a].start()

    time.sleep(5)

    out_concept_list = main_process_work(need_calc)

    print("begin to merge...")
    for key in concept_list.keys():
        if concept_list[key] == []:
            if key in out_concept_list.keys():
                concept_list[key] = out_concept_list[key]

    json.dump(concept_list, open("/data/wke18/data/ADV/Final/concept_v3.json", "w"))

    #get_CUI_info(ticket, "C0985346")
