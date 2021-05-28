#用于处理word pool，同时预处理同义词 pool
#要生成三个东西，word2id，word_neighbor，concept_neighbor，
from owlready2 import *
default_world.set_backend(filename = "/data/wke18/data/UMLS/pym.sqlite3")
PYM = get_ontology("http://PYM/").load()
CUI = PYM["CUI"]
import os
import json
import nltk
from gensim.models import KeyedVectors


file_dir = r"/data/wke18/data/ADV/Final/VALID"
word_list = {}
concept_list = {}
word2id = {}
id2word = {}
word_cnt = 0

def prepare_gensim():
    global model
    tmp_file = r"/data/wke18/WORD_VEC/test_word2vec.txt"
    model = KeyedVectors.load_word2vec_format(tmp_file)



def process_document():
    global word2id, id2word, word_cnt, concept_list, word_list, model
    #add UNK and PAD
    word2id['PAD'], word2id['UNK'] = 0, 1
    id2word[0], id2word[1] = 'PAD', 'UNK'
    word_cnt = 2
    #add words from files
    file_list = os.listdir(file_dir)
    for file in file_list:
        full_file = os.path.join(file_dir, file)
        data = json.load(open(full_file, "r"))
        text = nltk.word_tokenize(data["TEXT"])
        for word in text:
            try:
                temp = model[word]
            except:
                continue
            if word not in word2id.keys():
                word_list[word] = []
                word2id[word] = word_cnt
                id2word[word_cnt] = word
                word_cnt += 1
        for concept in data["MetaMap"]:
            if concept["CUI"] not in concept_list.keys():
                concept_list[concept["CUI"]] = []

def process_word_neighbor():
    global word2id, id2word, model, word_cnt, word_list
    for word in word_list.keys():
        neighbor_list = model.most_similar(positive=[word], topn=10)
        for neighbor in neighbor_list:
            word_list[word].append(neighbor)
            if neighbor[0] not in word2id.keys():
                word2id[neighbor[0]] = word_cnt
                id2word[word_cnt] = neighbor[0]
                word_cnt += 1

def process_concept_neighbor():
    global word2id, id2word, word_cnt, concept_list, word_cnt, model
    for concept in concept_list.keys():
        term = CUI[concept]
        synonyms = term.synonyms
        for syno in synonyms:
            syno_text = nltk.word_tokenize(syno)
            concept_list[concept].append(syno_text)
            for word in syno_text:
                try:
                    temp = model[word]
                except:
                    continue
                if word not in word2id.keys():
                    word2id[word] = word_cnt
                    id2word[word_cnt] = word
                    word_cnt += 1

    print("all word num:", word_cnt)

def output_data():
    output_file = r"/data/wke18/data/ADV/Final"
    json.dump(word2id, open(os.path.join(output_file, "word2id.json"), "w"))
    json.dump(id2word, open(os.path.join(output_file, "id2word.json"), "w"))
    json.dump(concept_list, open(os.path.join(output_file, "concept.json"), "w"))
    json.dump(word_list, open(os.path.join(output_file, "concept.json"), "w"))

if __name__ == "__main__":
    prepare_gensim()
    process_document()
    process_word_neighbor()
    process_concept_neighbor()
    output_data()
