import torch
import logging
import tqdm
import numpy as np
import os
from optim import Adam
from train_utils import set_lr
from torch.nn import CrossEntropyLoss


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
valid_logger = open('logger/dd_eval_log.txt','a+',buffering=1)
train_logger = open('logger/dd_train_log.txt','a+',buffering=1)
CACHE_EMPTY_STEP = 10000

def run(hparams, model, train_dataloader, valid_dataloader, device,out_dir='checkpoints'):
    learning_rate = hparams['learning_rate']
    accumulate_step = hparams['accumulate_step']
    lr_schedule = hparams['lr_schedule']
    warmup_steps = hparams['warmup_steps']
    warmup_proportion = hparams['warmup_proportion']
    n_embd = hparams['n_embd']
    num_optim_steps = hparams['num_optim_steps']
    train_batch_size = hparams['train_batch_size']
    valid_step = hparams['valid_step']
    no_token_id= hparams['no_token_id']

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Number of parameter = {}'.format(total_params))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln']
    optimizer_grouped_parameters = [
        {'params': [p for n,p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay':0.01},
        {'params': [p for n,p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = Adam(optimizer_grouped_parameters, learning_rate, max_grad_norm = 1.0)

    step = 0
    global_step = 0
    epoch = 0

    while True:
        model.train()
        (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0,0.0,0.0,0,0
        n_token_real, n_token_total = 0, 0
        pbar = tqdm.tqdm(enumerate(train_dataloader),total=len(train_dataloader))

        for i,batch in pbar:
            batch = tuple(t.cuda() for t in batch)
            input_ids, position_ids, token_type_ids, label_ids, *_ = batch
            if no_token_id:
                token_type_ids = None
            loss, ppl = model(input_ids, position_ids, token_type_ids, label_ids)
            loss = loss.mean()
            loss = loss / (train_batch_size/input_ids.shape[0])
            loss.backward()
            nb_tr_steps += 1
            tr_loss += float(loss.sum().item()) * (train_batch_size /input_ids.shape[0])

            if ppl.sum().item() < 1000000:
                tr_ppl += ppl.sum().item()
            else:
                tr_ppl += mean_ppl

            mean_loss = tr_loss / nb_tr_steps
            mean_ppl = tr_ppl / nb_tr_steps

            n_token_total += input_ids.shape[0]*input_ids.shape[1]
            n_token_real += (input_ids !=0).sum().item()

            #gradient update
            step += 1
            if step % accumulate_step == 0:
                set_lr(optimizer,global_step,lr_schedule,
                       learning_rate,warmup_steps,
                       warmup_proportion,n_embd,num_optim_steps)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                print('epoch: {}, global_step: {}, step: {}, mean_loss: {}, mean_ppl:{}'.format(
                    epoch + 1, global_step + 1, step + 1, mean_loss, mean_ppl),
                    file=train_logger)

                if global_step % valid_step == 0:
                    print('Saving model...')
                    torch.save({'model':model.state_dict(),
                                'epoch':epoch,
                                'hparams':hparams,},
                               os.path.join(out_dir,f'GPT2-pretrain-step-{global_step}.pkl'))
                    eval_loss,eval_ppl = valid(model,valid_dataloader,epoch,device)
                    print('{},{},{},{},{}'.format(
                        epoch + 1, global_step + 1, step + 1, eval_loss, eval_ppl),
                        file=valid_logger)
                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    model.train()
                if global_step >= num_optim_steps:
                    break
            if (step + 1) % CACHE_EMPTY_STEP == 0:
                torch.cuda.empty_cache()
        if global_step >= num_optim_steps:
            break
        epoch += 1
    train_logger.close()
    valid_logger.close()


def valid(model, valid_dataloader, epoch_id,device):
    logger.info('compute eval model loss, using eval mode,'
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    with torch.no_grad():
        for step,batch in enumerate(valid_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, position_ids, token_ids, label_ids = batch
            n_sample = input_ids.shape[0]
            loss,ppl = model(input_ids,position_ids,token_ids,label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)
