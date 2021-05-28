from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

# 输入文件
glove_file = datapath("/data/wke18/WORD_VEC/glove.42B.300d.txt")
# 输出文件
tmp_file = get_tmpfile("/data/wke18/WORD_VEC/test_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>

# 开始转换
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)

# 加载转化后的文件
model = KeyedVectors.load_word2vec_format(tmp_file)

#测试
for key in model.wv.similar_by_word('fever', topn =10):
    print(key)