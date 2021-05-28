from .nlp.BasicBert import BasicBert
from .nlp.LSTM import LSTM
from .nlp.TextCNN import TextCNN
from .nlp.CAML_NACCL import CAML
from .nlp.GRNN_HA import GRNNHA

model_list = {
    "BasicBert": BasicBert,
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "CAML": CAML,
    "GRNNHA": GRNNHA
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
