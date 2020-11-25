import logging
import tqdm
import torch
import os
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from data_loader import GPT2DataLoader,InputFeatures,GPT2FeatureDataset
from train_utils import load_model
from dialogGPT import GPT2LMHeadModel
from pytorch_pretrained_bert.modeling_gpt2 import  GPT2Config

logger = logging.getLogger('logger')


def test(model,tokenizer, dataset):
    model.eval()
    pbar = tqdm.tqdm(enumerate(dataset), total=len(dataset))
    for idx, data in pbar:
        data = tuple(t.cuda() for t in data)
        inference(data,model,tokenizer)
    print("Done !")

def inference(data,model,tokenizer):

    input_ids, position_ids, token_type_ids, label_ids, *_ = data
    k = input_ids[0].tolist().index(50256)


    target = input_ids[0][k+1:]

    logger.info("Context : {}".format(tokenizer.decode(input_ids[0][:k+1].tolist())))
    logger.info("Target: {} ".format(tokenizer.decode(target.tolist())))

    input_ids = input_ids[:,:k+1]

    with torch.no_grad():
        output = []
        while True:
            logits, presents = model(input_ids)
            uttr = torch.argmax(logits,dim=2)
            output.append(uttr[:,-1].item())
            if uttr[0][-1].item() == 50256:
                break
            k += 1
            if k < len(input_ids[0]):
                input_ids[:,k] = uttr[0][-1].item()
            else:
                if k >=512:
                    break
                input_ids = torch.cat((input_ids[0],uttr[0][-1].unsqueeze(dim=0)))
                input_ids = input_ids.view(1,-1)
    logger.info("Prediction: {}".format(tokenizer.decode(output)))
    return logits, tokenizer.decode(output[:-1])

#def chat_inference(data,model,tokenizer):
#    input_ids, position_ids, token_type_ids, label_ids, *_ = data
#    k = input_ids[0].tolist().index(50256)
#    input_ids = input_ids[:, :k + 1]
#    with torch.no_grad():
#        output = []
#        while True:
#            logits, presents = model(input_ids)
#            logits = top_k_sampling(logits[0,-1.:].squeeze())
#            uttr = torch.argmax(logits)
#            output.append(uttr.item())
#            if uttr.item() == 50256:
#                break
#            k += 1
#            if k < len(input_ids[0]):
#               input_ids[:, k] = uttr.item()
#            else:
#                if k >= 512:
#                    break
#                input_ids = torch.cat((input_ids[0], uttr.unsqueeze(dim=0)))
#                input_ids = input_ids.view(1, -1)
#    logger.info("Prediction: {}".format(tokenizer.decode(output)))
#   return logits, tokenizer.decode(output[:-1])

def chat_inference(model,conditioned_tokens,generated_tokens):
    indexed_tokens = conditioned_tokens + generated_tokens
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    logits = predictions[0,-1,:]
    filtered_logit = top_k_sampling(logits)
    probabilities = F.softmax(filtered_logit,dim=-1)
    next_token = torch.multinomial(probabilities,1)
    generated_tokens.append(next_token.item())
    return next_token.item()

def chat(model,tokenizer):
    model.eval()
    conditioned_tokens = []
    generated_tokens = []

    while True:
        text = input("> ")
        print("User: {}".format(text))

        if text == ":q" or text == ":quit":
            break
        conditioned_tokens += tokenizer.encode(text) + [50256]
        print(tokenizer.decode(conditioned_tokens))
        while True:
            result = chat_inference(model,conditioned_tokens,generated_tokens)

            if result == 50256:
                print("Bot: {}".format(tokenizer.decode(generated_tokens[:-1])))
                conditioned_tokens += generated_tokens
                generated_tokens = []
                break

#def chat(model,tokenizer):
#    model.eval()
#    src_list = ''
#    while(1):
#        src_sent = input("> ")
#        print("User: {}".format(src_sent))
#        if src_sent == ":q" or src_sent == ":quit":
#            break
#        if len(src_list) == 0:
#            src_list = src_sent
#        else:
#            src_list = src_list + ' ' +  src_sent##

#        input_ids = torch.tensor(tokenizer.encode(src_list+"<|endoftext|>"),dtype=torch.long).view(1,-1).cuda()

 #       data = (input_ids, None, None, None)
 #       _, response = chat_inference(data,model,tokenizer)

 #       print("Bot : {}".format(response))
 #       src_list = src_list + " "+ response[:-1]


def top_k_sampling(logits,top_p=0.9,filter_value=-float('inf')):
    assert logits.dim() == 1
    sorted_logits, sorted_indices = torch.sort(logits,descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits,dim=-1),dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
    sorted_indices_to_remove[...,0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_size = 'middle'
    if model_size == 'small':
        config_path = '117M/config.json'
    elif model_size == 'middle':
        config_path = '345M/config.json'
    elif model_size == 'big':
        config_path = '762M/config.json'
    config = GPT2Config.from_json_file(os.path.join('./configs/', config_path))
    print(config)
    checkpoint_path ="checkpoints/medium_ft.pkl"#'Cornell_models/GPT_Cornell_models.pkl' #"checkpoints/medium_fs.pkl"
    model = load_model(GPT2LMHeadModel(config), checkpoint_path,test=False)
    model = model.to(device)

   # train_data_loader = GPT2DataLoader(data_path='DailyDialog/train_text.txt',
   #                                    vocab_file='./vocab_file/encoder.json',
   #                                    bpe_merges='vocab_file/merges.txt',
   #                                    bucket=2,
   #                                    batch_size=5,
   #                                    max_seq_len=512)
    vocab_file = './configs/345M/vocab.json'
    bpe_merges = './configs/345M/merges.txt'
    #valid_data_loader = GPT2DataLoader(data_path='DailyDialog/test_text.txt',
    #                                   vocab_file=vocab_file,
    #                                   bpe_merges=bpe_merges,
    #                                   bucket=2,
    #                                   batch_size=1,
    #                                   max_seq_len=512)
    hparams = {'learning_rate': 1e-5,
               'accumulate_step': 2,
               'lr_schedule': 'noam',
               'warmup_steps': 16000,
               'warmup_proportion': 0.1,
               'n_embd': 768,
               'num_optim_steps': 100000,
               'train_batch_size': 1,
               'valid_step': 10000,
               'device':device,
               'vocab_file':vocab_file,
               'bpe_merge':bpe_merges,
               'beam_width':1,
               'max_len':1024}
    tokenizer = GPT2Tokenizer(hparams['vocab_file'], hparams['bpe_merge'])
    #test(hparams,model,valid_data_loader)
    chat(model,tokenizer,device)











