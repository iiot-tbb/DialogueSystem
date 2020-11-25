import logging
import os
import torch
from transformers import GPT2Tokenizer
from train_utils import load_model
from dialogGPT import GPT2LMHeadModel
from data_loader import GPT2DataLoader, InputFeatures,GPT2FeatureDataset
from pytorch_pretrained_bert.modeling_gpt2 import  GPT2Config

def create_log(logfilename):
    logger = logging.getLogger('logger')
    format = '%(message)s'
    handler = logging.FileHandler(filename=logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def run_test(model_dir,data_dir,mode,config_path='345M/',beam_width = 10):
    config_path = config_path + 'config.json'
    vocab_path = config_path + 'vocab.json'
    merge_path = config_path + 'merges.txt'
    checkpoint_path = model_dir + '/GPT_model.pkl'
    log_filename = model_dir + '/test_data.log'

    config = GPT2Config.from_json_file(os.path.join('./configs/',config_path))

    create_log(log_filename)
    print("Building model")
    model = load_model(GPT2LMHeadModel(config),checkpoint_path,test=True).cuda()
    model.eval()
    tokenizer = GPT2Tokenizer(vocab_path,merge_path)
    if mode == 'test':
        print('Loading test dataset...')
        test_data_loader = GPT2DataLoader(data_path = data_dir,
                                          vocab_file = vocab_path,
                                          bpe_merges=merge_path,
                                          bucket=2,
                                          batch_size=1,
                                          max_seq_len=512)



