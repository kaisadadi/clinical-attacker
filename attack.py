#deepfool implementation
import os
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from model import get_model
import argparse
from config_parser import create_config
from reader.reader import init_dataset, init_formatter, init_test_dataset
#from attack_zoo import deepfool, show_important_words, fast_gradient, clinical_fool, FGSMNNS, clinical_fool_black
from attack_zoo import clinical_fool, clinical_fool_black
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

model_id = 32

if __name__ == "__main__":
    #pre-preparation
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", default="config/nlp/TextCNN.config")
    parser.add_argument('--gpu', '-g', help="gpu id list", default="0")
    args = parser.parse_args()

    configFilePath = args.config

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    os.system("clear")
    logger.info("Begin to initialize models...")
    config = create_config(configFilePath)
    model = get_model(config.get("model", "model_name"))(config, gpu_list)
    
    parameter_path = config.get("output", "model_path") + "/" + config.get("output", "model_name") + "/" + str(model_id) + ".pkl"
    parameters = torch.load(parameter_path)
    model.load_state_dict(parameters["model"])
    model = model.cuda()

    logger.info("Begin to initialize datasets and formatter...")

    init_formatter(config, ["test"])
    test_dataset = init_test_dataset(config)


    #start to work
    #show_important_words(model = model, dataset = test_dataset)
    #fast_gradient(config=config, model = model, dataset = test_dataset)

    #clinical_fool
    fooler = clinical_fool(config)
    #fooler = FGSMNNS(config)
    #fooler.attack(model, test_dataset)
    fooler.attack(model, test_dataset)
