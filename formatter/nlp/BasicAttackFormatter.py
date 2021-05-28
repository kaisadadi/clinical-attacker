import json
import torch
import numpy as np

from formatter.Basic import BasicFormatter
import jieba.posseg as posseg


class Basic_Attack_Formatter(BasicFormatter):
    #分词，word2id，label，UNK和PAD的处理
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.word2id_path = config.get("data", "word2id_path")
        self.word2id = json.load(open(self.word2id_path, "r"))

        self.max_len = config.getint("data", "max_seq_length")
        self.output_dim = config.getint("model", "output_dim")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        input = []
        label = []
        property = []
        medical_pos = []
        CUI = []
        length = []
        raw_texts = []

        for item in data:
            process_res = posseg.cut(item["text"])
            self.begin_points = {} #统计所有起点
            length_cnt = 0
            TEXT, PROPERTY = [], []
            RAWTEXT = []
            for idx, piece_res in enumerate(process_res):
                TEXT.append(piece_res.word)
                RAWTEXT.append(piece_res.word)
                PROPERTY.append(piece_res.flag)
                #处理端点
                self.begin_points[length_cnt] = idx
                length_cnt += len(piece_res.word)

            length.append(len(TEXT))
            raw_texts.append(TEXT)
            token = []
            pos_tag = []
            for word in TEXT:
                word = word.lower()
                if word in self.word2id.keys():
                    token.append(self.word2id[word])
                else:
                    token.append(1)  #UNK 1
            #词性
            for tag in PROPERTY:
                if tag in ['a', 'ad', 'an', 'Ag']:
                    pos_tag.append(1)
                elif tag in ['dg', 'd']:
                    pos_tag.append(0)
                else:
                    pos_tag.append(-1)

            while len(token) < self.max_len:
                token.append(0)
                pos_tag.append(-1)
            token = token[0:self.max_len]
            pos_tag = pos_tag[0:self.max_len]

            input.append(token)
            item_label = [0] * self.output_dim
            for disease in item["label"]:
                item_label[disease] = 1

            label.append(item_label)
            property.append(pos_tag)

            item_medical_pos, item_CUI = self.process_entity(item["entity"], RAWTEXT)
            medical_pos.append(item_medical_pos)
            CUI.append(item_CUI)


        input = torch.from_numpy(np.array(input)).long()
        label = torch.from_numpy(np.array(label)).long()
        property = np.array(property)
        return {'input': input, 'label': label, "property": property, "medical_pos": medical_pos, "CUI": CUI, "length": length, "rawtext": raw_texts}
    

    def process_entity(self, entities, text):
        #生成medical_pos和CUI，每个位置对应的CUI以最后一次覆盖为准
        medical_pos = {}
        CUI = {}
        for item in entities:
            start_pos, end_pos = item["left"], item["right"]   #左闭右开
            if (start_pos in self.begin_points.keys()) and (end_pos in self.begin_points.keys()):
                left = self.begin_points[start_pos]
                right = self.begin_points[end_pos]
                CUI[item["CUI"]] = [left, right - 1]
                for idx in range(left, right):
                    medical_pos[idx] = item["CUI"]
        return medical_pos, CUI

