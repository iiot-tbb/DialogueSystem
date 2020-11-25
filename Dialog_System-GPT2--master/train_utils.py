import logging
import os
import torch
from optim import noam_decay, noamwd_decay, warmup_linear

logger = logging.getLogger(__name__)

def load_model(model, checkpoint, verbose = False, test= False):
   # n_gpu = args.n_gpu
   # device = args.device
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s !'% model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        if test:
            model_state_dict = torch.load(checkpoint)['model']
        else:
            model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model,"transformer") and all(not s.startswith("transformer.")
            for s in model_state_dict.keys())):
            logger.info("loading transformer only")
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

      #  if args.fp16:
       #     logger.info("in fp16, model.half() activated")
       #     model.half()
      #  model.to(device)
      #  if n_gpu > 1:
      #      logging.info("data parallel because more than one gpu")
      #      model = torch.nn.DataParallel(model)
    return model

def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t in model_state_dict:
            new_key = t
            if t.startswith('module.'):
                new_key = t.replace('module.','')
            old_keys.append(t)
            new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

def set_lr(optimizer, step, schedule, lr, warmup_steps, warmup_proportion,n_embd,tot_steps):
    if schedule == 'None':
        lr_this_step = lr
    elif schedule == 'noam': #transformer like
        lr_this_step = lr * 1e4 * noam_decay(step+1,warmup_steps,n_embd)
    elif schedule == 'noamwd':
        lr_this_step = lr * 1e4 * noamwd_decay(step+1,warmup_steps,n_embd)
    else:
        lr_this_step = lr * warmup_linear(step / tot_steps,
                                          warmup_proportion)
    for param_group in optimizer.param_groups:
        param_group['lr'] =lr_this_step





