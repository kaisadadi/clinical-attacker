import json
import os
import jieba
from gensim.models import KeyedVectors

filtered_train_dir = r"/data2/wk/data/pot/train_data_20"
corpus_path = r"/home/wke18/Changgeng-Hospital/Data/embedding/Tencent_AILab_ChineseEmbedding.txt"

word_dict = {}
model = KeyedVectors.load_word2vec_format(corpus_path, binary=False)

def gen_word_list(train_data_dir = None):
    #生成数据中的词集合
    global word_dict
    file_list = os.listdir(train_data_dir)
    for file in file_list:
        file = os.path.join(train_data_dir, file)
        data = json.load(open(file, "r"))
        text = jieba.lcut(data["text"])
        for word in text:
            if word in word_dict.keys():
                word_dict[word] = 0
            word_dict[word] += 1
        
def gen_entity_pool_list():
    #生成实体替换池的词集合
    pass

def gen_similarity_word_list():
    pass

if __name__ == "__main__":
    #gen_word_list(filtered_train_dir)
    print(model.word_vec('地球'))
    print(model.most_similar('酒杯'))