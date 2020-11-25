
from train_utils import load_model
from dialogGPT import GPT2LMHeadModel
from pytorch_pretrained_bert.modeling_gpt2 import  GPT2Config
from data_loader import GPT2DataLoader
from train import run
import os
import torch


if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_size = 'medium'
    if model_size == 'small':
        config_path = '117M/config.json'
        vocab_path = '117M/vocab.json'
        merges_path = '117M/merges.txt'
    elif model_size == 'medium':
        config_path = '345M/config.json'
        vocab_path = '345M/vocab.json'
        merges_path = '345M/merges.txt'
    elif model_size == 'large':
        config_path = '762M/config.json'
        vocab_path = '762M/vocab.json'
        merges_path = '762M/merges.txt'

    config = GPT2Config.from_json_file(os.path.join('./configs/',config_path))
    model = load_model(GPT2LMHeadModel(config),"checkpoints/medium_ft.pkl")
    device = range(torch.cuda.device_count())
    model = torch.nn.DataParallel(model,device_ids=device).cuda()



    vocab_file = os.path.join('./configs/',vocab_path)
    bpe_merges = os.path.join('./configs/',merges_path)

    train_data_loader = GPT2DataLoader(data_path='Data/Cornell_movie_dialogs/dd_train.txt',
                                 vocab_file=vocab_file,
                                 bpe_merges=bpe_merges,
                                 bucket=2,
                                 batch_size=2,
                                 max_seq_len=512)

    valid_data_loader = GPT2DataLoader(data_path = 'Data/Cornell_movie_dialogs/dd_validation.txt',
                                       vocab_file=vocab_file,
                                       bpe_merges=bpe_merges,
                                       bucket=2,
                                       batch_size=2,
                                       max_seq_len=512)
    hparams = {'learning_rate':1e-5,
               'accumulate_step':4,
               'lr_schedule':'noam',
               'warmup_steps':16000,
               'warmup_proportion':0.1,
               'n_embd':1024,
               'num_optim_steps':100000,
               'train_batch_size':2,
               'valid_step':10000,
               'no_token_id':True}
    run(hparams, model, train_data_loader,valid_data_loader,device)


