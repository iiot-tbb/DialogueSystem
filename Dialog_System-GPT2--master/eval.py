from dialogGPT import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from train_utils import load_model
from data_loader import GPT2DataLoader
#from transformers import GPT2Config
from pytorch_pretrained_bert.modeling_gpt2 import  GPT2Config
from eval_utils import gpt_chat
import sys
import os
import torch
import logging
#sys.path.append('/export/home2/NoCsBack/hci/mingxiao/DialogueGPT2/bertviz')
sys.path.append('/Users/limingxiao/Desktop/Pycharm_projects/bertviz/bertviz')
#from bertviz import model_view

model_name = 'Ubuntu'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
log_file = model_name +'_eval.log'
log_level = logging.INFO
logging.basicConfig(filename=log_file,level=log_level)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = './configs/345M/config.json'
config = GPT2Config.from_json_file(config_path)
#checkpoint_path ='Ubuntu_checkpoints/GPT2-pretrain-step-10000.pkl'
checkpoint_path = './checkpoints/medium_ft.pkl'
model = load_model(GPT2LMHeadModel(config),checkpoint_path,test=False)
model = model.cuda()

test_data_file = './Data/' + model_name +'/new_test_.txt'
test_data = GPT2DataLoader.read_data(test_data_file)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print('Chat to bot.')
dial_his = []
while True:
    sent = input('You : > ')
    if sent == 'quit' or sent == 'q':
        break
    dial_his.append(sent)
    bot_res = gpt_chat(model,dial_his,tokenizer,sampling=True)
    bot_res = tokenizer.decode(bot_res)
    print('Bot :> {}'.format(bot_res))
    dial_his.append(bot_res)

# test model on test set and write results to a file
#for i,dial in enumerate(test_data):
#    print(i)
#    dial_his = dial[:-1]
#    tgt = dial[-1]
#    logger.info('Test data {}'.format(i))
#    logger.info('Context : {}'.format(' '.join(dial_his)))
#    logger.info('Tgt : {}'.format(dial[-1]))
#    response = gpt_chat(model,dial_his,tokenizer)
#    logger.info('Responses: {}'.format(tokenizer.decode(response)))
#print("Done ! ! !")


