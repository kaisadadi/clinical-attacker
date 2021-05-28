import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import json
import pickle
import os
import numpy as np
import nltk
from tools.accuracy_tool import gen_micro_macro_result

class clinical_fool_black():
    #multi-label clinical fool
    def __init__(self, config):
        self.w2v_path = config.get("data", "w2v_path")
        self.word2id_path = config.get("data", "word2id_path")
        self.id2word_path = config.get("data", "id2word_path")
        self.word_neighbor_path = config.get("data", "word_neighbor_path")
        self.concept_path = config.get("data", "concept_path")
        self.w2v = pickle.load(open(self.w2v_path, "rb"))
        self.word2id = json.load(open(self.word2id_path, "r"))
        self.id2word = json.load(open(self.id2word_path, "r"))
        self.word_neighbor = json.load(open(self.word_neighbor_path, "r"))
        self.concept = json.load(open(self.concept_path, "r"))
        self.lam = config.getfloat("attack", "lam")
        self.k = config.getint("attack", "k")
        self.max_len = config.getint("data", "max_seq_length")
        self.task = config.getint("attack", "task")
        self.gen_result = True
        self.output_dir = ".json"

    def attack(self, model, dataset):
        #要输出SR，PR(记录一下总次数)，L2 sum(记录一下)，mic和mac(维护一个F1)
        Perturbation_num = 0
        L2_sum = 0
        F1 = []
        for a in range(10):
            F1.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        #lam是Perturbation Distance Score的权重
        model.eval()   #wc这句话不加坑死我了
        success = 0
        fail = 0
        out_data = []
        for dataset_idx, data in enumerate(dataset):
            if dataset_idx >= 500:
                break
            if self.gen_result ==True and  len(out_data) >= 100:
                break
            text_length = data["length"]
            target_label = Variable(1 - data["label"]).cuda().float()
            label = Variable(data["label"]).cuda().float()
            input_seq = data["input"][0].numpy().tolist()
            property_seq = data["property"][0].tolist()
            medical_pos = data["medical_pos"][0]
            CUI_pos = data["CUI"][0]
            iter_num = 0
            iter_L2_sum = 0
            while True:
                iter_num += 1
                if iter_num >= 32:
                    print("fail for exceed iteration num!")
                    fail += 1
                    break
                #localize k candidates
                if len(input_seq) == 0:
                    print("shit #1")
                result = model({"input": Variable(torch.from_numpy(np.array(input_seq)).unsqueeze(0)).cuda().long()})
                old_logits, prediction = result["logits"], result["prediction"]
                #print(old_logits)
                if self.check_leave(prediction, label, self.task):
                    #完成全部修改，出口一
                    #print("exit #1")
                    success += 1
                    Perturbation_num += float(iter_num / text_length[0])
                    L2_sum += iter_L2_sum
                    if self.gen_result:
                        out_data.append({"input": input_seq, "label": data["label"].numpy()})
                    print("success with %d times!" %(iter_num - 1))
                    break
                loss = self.calc_loss(old_logits, target_label)[0]
                top_k = self.black_box(model, input_seq, self.k, label)
                candidate = []  #{pos, oldvec, newvec, type}, pos指的是修改的起点
                #填充candidate
                candi_flag = 0
                for k in range(self.k):
                    location = top_k[k]
                    #print("location: ", location)
                    #先判断是不是医学词汇
                    if location in medical_pos.keys():
                        try:
                            CUI = medical_pos[location]
                            start_pos, end_pos = CUI_pos[CUI]
                        except:
                            candi_flag = 1
                            break
                        try:
                            oldvec = [input_seq[idx] for idx in range(start_pos, end_pos + 1)]
                        except:
                            continue
                        for atom in self.concept[CUI]:
                            atom = nltk.word_tokenize(atom)
                            newvec = []
                            for atom_word in atom:
                                if atom_word in self.word2id.keys():
                                    newvec.append(self.word2id[atom_word])
                                else:
                                    newvec.append(self.word2id["UNK"])
                            candidate.append({"pos": start_pos, "oldvec": oldvec, "newvec": newvec, "type": "med"})
                    else:
                        #先进行替换，如果是副词考虑删去
                        word = self.id2word[str(input_seq[location])]
                        if word in self.word_neighbor.keys():
                            for neighbor in self.word_neighbor[word]:
                                neighbor_id = self.word2id[neighbor]
                                candidate.append({"pos": location, "oldvec": [input_seq[location]], "newvec": [neighbor_id], "type": "rep"})
                        #删去副词的操作
                        if property_seq[location] == 0:
                            if location > 0 and property_seq[location - 1] == 1:
                                old_vec = input_seq[location - 1 : location + 1]
                                new_vec = input_seq[location - 1]
                                candidate.append({"pos": location - 1, "oldvec": old_vec, "newvec": new_vec, "type": "rem"})
                            elif location < self.max_len - 1 and property_seq[location + 1] == 1:
                                old_vec = input_seq[location: location + 2]
                                new_vec = input_seq[location + 1]
                                candidate.append({"pos": location, "oldvec": old_vec, "newvec": new_vec, "type": "rem"})
                    if candi_flag:
                        break
                #candidate组成batch来跑
                candidate_input = []
                for candi in candidate:
                    pos = candi["pos"]
                    oldvec = candi["oldvec"]
                    newvec = candi["newvec"]
                    new_seq = input_seq[:pos] + newvec + input_seq[pos + len(oldvec):]
                    while len(new_seq) < self.max_len:
                        new_seq.append(0)
                    new_seq = new_seq[:self.max_len]
                    candidate_input.append(new_seq)
                if len(candidate_input) == 0:
                    # 负分，出口#2
                    fail += 1
                    print("fail for no candidate!")
                    break
                if len(candidate_input) > 256:
                    candidate_input = candidate_input[:256]
                candidate_input = Variable(torch.from_numpy(np.array(candidate_input))).cuda().long()
                logits = model({"input": candidate_input})["logits"]
                candidate_loss = self.calc_loss(logits, target_label).detach().cpu().numpy()
                del candidate_input, logits
                #print("candidate_loss:", np.min(candidate_loss))
                #print("candidate_idx:", np.argmin(candidate_loss))
                #print("candidate_logits:", logits[np.argmin(candidate_loss)])
                candidate_score = []
                for idx in range(min(len(candidate), 200)):
                    candidate_score.append(self.calc_score(loss.item(), candidate_loss[idx], candidate[idx]["oldvec"], candidate[idx]["newvec"]))
                #选取最高的结果
                highest_candidate = candidate_score.index(max(candidate_score))
                #print("highest score:", candidate_score[highest_candidate])
                #print("highest idx:", highest_candidate)
                if candidate_score[highest_candidate] <= 0:
                    #负分，出口#2
                    fail += 1
                    print("fail for all gg!")
                    #print("minus score:", candidate_score[highest_candidate])
                    #print("exit #2")
                    break
                #更新input_seq, property_seq, medical_pos, CUI_pos
                pos, oldvec = candidate[highest_candidate]["pos"], candidate[highest_candidate]["oldvec"]
                type, newvec = candidate[highest_candidate]["type"], candidate[highest_candidate]["newvec"]
                iter_L2_sum += np.linalg.norm(self.get_avg_vec(oldvec) - self.get_avg_vec(newvec))
                new_input_seq = input_seq[:pos] + newvec + input_seq[pos + len(oldvec):]
                while len(new_input_seq) < self.max_len:
                    new_input_seq.append(0)
                new_input_seq = new_input_seq[:self.max_len]
                input_seq = new_input_seq
                if type == "med":
                    property_seq = property_seq[:pos] + [0] * len(newvec) + property_seq[pos + len(oldvec):]
                elif type == "rep":
                    property_seq = property_seq[:pos] + [property_seq[pos]] + property_seq[pos + len(oldvec):]
                elif type == "rem":
                    property_seq = property_seq[:pos] + [1] + property_seq[pos + len(oldvec):]
                while len(property_seq) < self.max_len:
                    property_seq.append(0)
                property_seq = property_seq[:self.max_len]
                new_medical_pos = {}
                for key in medical_pos.keys():
                    if key < pos:
                        new_medical_pos[key] = medical_pos[key]
                    elif key == pos:
                        if type != "med":
                            continue
                        for key_idx in range(len(newvec)):
                            new_medical_pos[key + key_idx] = medical_pos[key]
                    else:
                        new_medical_pos[key + len(newvec) - len(oldvec)] = medical_pos[key]
                new_CUI_pos = {}
                break_flag = 0
                for key in CUI_pos.keys():
                    start_pos, end_pos = CUI_pos[key]
                    if end_pos < pos:
                        new_CUI_pos[key] = [start_pos, end_pos]
                    elif start_pos == pos and type == "med":
                        new_CUI_pos[key] = [pos, pos + len(newvec) - 1]
                    elif start_pos > pos:
                        bias = len(newvec) - len(oldvec)
                        new_CUI_pos[key] = [start_pos + bias, end_pos + bias]
                    else:
                        print(start_pos, end_pos)
                        print(pos, type, oldvec, newvec)
                        print(key, CUI_pos[key])
                        break_flag = 1
                        break
                    if break_flag:
                        break
                medical_pos = new_medical_pos
                CUI_pos = new_CUI_pos
            #print("finish one data!")
            #再跑一个最终结果
            output = model({"input": Variable(torch.from_numpy(np.array(input_seq))).cuda(), "label": label})
            temp_F1 = output["acc_result"]
            for a in range(10):
                F1[a]["TP"] += temp_F1[a]["TP"]
                F1[a]["FP"] += temp_F1[a]["FP"]
                F1[a]["TN"] += temp_F1[a]["TN"]
                F1[a]["FN"] += temp_F1[a]["FN"]
            if (dataset_idx+1) % 100 == 0:
                SR = float(success / (dataset_idx+1))
                L2_sum = float(L2_sum / (dataset_idx+1))
                Perturbation_num = float(Perturbation_num / (dataset_idx+1))
                F1_result = gen_micro_macro_result(F1)
                macro_F1 = F1_result["macro_f1"]
                mirco_F1 = F1_result["micro_f1"]
                print("------------------------------------------------------")
                print("idx", dataset_idx + 1)
                print("SR:", SR)
                print("L2_num", L2_sum)
                print("PR", Perturbation_num)
                print("macro", macro_F1)
                print("micro", mirco_F1)
                print("fail num:", fail)
                print("------------------------------------------------------")
        SR = float(success / 500)
        L2_sum = float(L2_sum / 500)
        Perturbation_num = float(Perturbation_num / 500)
        F1_result = gen_micro_macro_result(F1)
        macro_F1 = F1_result["macro_f1"]
        mirco_F1 = F1_result["micro_f1"]

        print("SR:", SR)
        print("L2_num", L2_sum)
        print("PR", Perturbation_num)
        print("macro", macro_F1)
        print("micro", mirco_F1)
        print("fail num:", fail)

        json.dump(out_data, open(self.output_dir, "r"))


    def calc_loss(self, logits, labels):
        #用于loss的计算，考虑batch个样本
        #relu-like optimization function
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)
            labels = labels.repeat(int(logits.shape[0] / labels.shape[0]), 1)
        pre_loss = torch.abs(logits - labels)
        pre_loss = (1 - torch.ge(pre_loss, 0.5)).float() * 0.4 * pre_loss + torch.ge(pre_loss, 0.5).float() * (1.6 * pre_loss - 0.6)
        loss = torch.sum(pre_loss, dim = 1)
        return loss

    def calc_score(self, old_loss, candidate_loss, old_vec, new_vec):
        # Perturbation Saliency Score,  Perturbation Distance Score(L2 norm)
        #old_vec和new_vec是list，里面存的编号
        saliency = self.calc_saliency_score(old_loss, candidate_loss)
        distance = np.linalg.norm(self.get_avg_vec(old_vec) - self.get_avg_vec(new_vec))
        return saliency - self.lam * distance


    def calc_saliency_score(self, old_loss, candidate_loss):
        #loss做差
        return old_loss - candidate_loss


    def get_avg_vec(self, vec_list):
        #计算平均的词向量
        avg_vec = []
        for vec in vec_list:
            avg_vec.append(self.w2v[vec])
        avg_vec = np.mean(np.array(avg_vec), axis=0)
        return avg_vec

    def check_leave(self, prediction, label, task):
        prediction = prediction.squeeze().detach().cpu().numpy()
        label = label.squeeze().detach().cpu().numpy()
        #print(prediction)
        #print(label)
        if np.sum(prediction != label) >= task:
            return True
        else:
            return False

    def black_box(self, model, input_seq, k, label):
        #black-box candidate selection
        top_k = []
        now_seq = input_seq.copy()
        while len(top_k) < k:
            left, right = 0, len(input_seq)
            origin_result = model({"input": Variable(torch.from_numpy(np.array(now_seq))).cuda()})
            score_origin = self.calc_loss(origin_result["logits"], label).detach().cpu().numpy()
            del origin_result
            while left != right:
                mid = int((left + right)/ 2)
                x_left = self.mask(now_seq, left, mid)
                x_right = self.mask(now_seq, mid+1, right)
                result = model({"input": Variable(torch.from_numpy(np.array([x_left, x_right]))).cuda()})
                logits = result["logits"]
                score_left, score_right = self.calc_loss(logits[0], label).detach().cpu().numpy(), self.calc_loss(logits[1], label).detach().cpu().numpy()
                del result, logits
                if abs(score_left - score_origin) > abs(score_right - score_origin):
                    right = mid
                else:
                    left = mid + 1
            top_k.append(left)
            now_seq[left] = 0
        return top_k


    def mask(self, input_seq, begin, end):
        #闭区间
        x = []
        for idx in range(len(input_seq)):
            if idx >= begin and idx <= end:
                x.append(0)
            else:
                x.append(input_seq[idx])
        return x


if __name__ == "__main__":
    pass