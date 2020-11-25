
from train_utils import load_model
from dialogGPT import GPT2LMHeadModel
from pytorch_pretrained_bert.modeling_gpt2 import  GPT2Config
from data_loader import GPT2DataLoader
from train import run
import os
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_size = 'small'
    if model_size == 'small':
        config_path = '117M/config.json'
    elif model_size == 'middle':
        config_path = '345M/config.json'
    elif model_size == 'big':
        config_path = '762M/config.json'
    config = GPT2Config.from_json_file(os.path.join('./configs/',config_path))
    model = load_model(GPT2LMHeadModel(config),"checkpoints/small_fs.pkl")
    model = model.to(device)

    train_data_loader = GPT2DataLoader(data_path='DailyDialog/train_text.txt',
                                 vocab_file='./vocab_file/encoder.json',
                                 bpe_merges='vocab_file/merges.txt',
                                 bucket=2,
                                 batch_size=5,
                                 max_seq_len=512)

    valid_data_loader = GPT2DataLoader(data_path = 'DailyDialog/test_text.txt',
                                       vocab_file='./vocab_file/encoder.json',
                                       bpe_merges='vocab_file/merges.txt',
                                       bucket=2,
                                       batch_size=5,
                                       max_seq_len=512)
    hparams = {'learning_rate':1e-5,
               'accumulate_step':2,
               'lr_schedule':'noam',
               'warmup_steps':16000,
               'warmup_proportion':0.1,
               'n_embd':768,
               'num_optim_steps':100000,
               'train_batch_size':5,
               'valid_step':10000}
    run(hparams, model, train_data_loader,valid_data_loader,device)


