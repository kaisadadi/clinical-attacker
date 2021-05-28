import os
import json
import torch
import torch.nn as nn
from model import get_model
import argparse
from config_parser import create_config
from formatter import Basic_Attack_Formatter
from attack_zoo import attack_white_demo, attack_black_demo

class attack_gate:
    #attack类
    def __init__(self, params):
        #params dict {"config": config file, "model_id": xx, "mode": "white" or "black", "gpu": gpu_id(多个的话,分开)}
        #prepare env
        self.mode = params.get("mode", "white")
        configFilePath = params.get("config", None)
        use_gpu = True
        gpu_list = []
        if params.get("gpu", None) is None:
            use_gpu = False
        else:
            use_gpu = True
            os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu"]
            device_list = params["gpu"].split(",")
            for a in range(0, len(device_list)):
                gpu_list.append(int(a))

        self.config = create_config(configFilePath)

        #init model and load parameters
        self.model = get_model(self.config.get("model", "model_name"))(self.config, gpu_list)
        model_id = params.get("model_id", -1)
        parameter_path = self.config.get("output", "model_path") + "/" + self.config.get("output", "model_name") + "/" + str(model_id) + ".pkl"
        parameters = torch.load(parameter_path)
        self.model.load_state_dict(parameters["model"])
        if use_gpu:
            self.model = self.model.cuda()

        #init formatter
        self.formatter = Basic_Attack_Formatter(self.config, "test")

        #init fooler
        if self.mode == "white":
            self.fooler = attack_white_demo(self.config)
        elif self.mode == "black":
            self.fooler = attack_black_demo(self.config)

    def attack(self, notes):
        #notes list of json files, {"text": "", "label": []}
        attack_res = self.fooler.attack(self.model, notes, self.formatter)
        for item in attack_res:
            print(item)
            print()

if __name__ == "__main__":
    #load data
    data_dir = r"/data2/wk/data/pot/test_data_20"
    filelist = os.listdir(data_dir)
    notes = []
    for file in filelist:
        file = os.path.join(data_dir, file)
        data = json.load(open(file, "r"))
        if "text" not in data.keys():
            data["text"] = data["note"]
            del data["note"]
        assert "label" in data.keys()
        notes.append(data)
        if len(notes) >= 10:
            break

    #init attacker
    params = {
        "config": "config/nlp/LSTM.config",
        "model_id": 16,
        "mode": "white",
        "gpu": "0"
    }
    attacker = attack_gate(params)

    #get res
    attacker.attack(notes)
