import torch
import sys
import torch.nn.functional as F
#sys.path.append('/export/home2/NoCsBack/hci/mingxiao/DialogueGPT2/bertviz')
sys.path.append('/Users/limingxiao/Desktop/Pycharm_projects/bertviz/bertviz')
#from bertviz import model_view


def gpt_chat(model,dia_list,tokenizer,sampling=True):
    model.eval()
    dia_his = ' '.join(dia_list)
    conditioned_tokens = tokenizer.encode(dia_his)+[50256]
    dia_list[-1] = dia_list[-1]+' <|endoftext|>'
    generated_tokens = []
    while True:
        result,attention = chat_inference(model,conditioned_tokens,generated_tokens,sampling)
        tokens = conditioned_tokens + generated_tokens
        tokens = tokenizer.convert_ids_to_tokens(tokens)
        #for i in range(24):
        #    model_view(attention[i], tokens[:-1])
        if result == 50256:
            return generated_tokens


def chat_inference(model,conditioned_tokens,generated_tokens,sampling = True):
    indexed_tokens = conditioned_tokens + generated_tokens
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.cuda()
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        attention = outputs[-1]
    logits = predictions[0,-1,:]
    if sampling:
        filtered_logit = top_k_top_p_filtering(logits,top_k=40,top_p=0.9)
        probabilities = F.softmax(filtered_logit,dim=-1)
        next_token = torch.multinomial(probabilities,1)
        generated_tokens.append(next_token.item())
        return next_token.item(),attention
    next_token = torch.argmax(logits,dim=-1)
    generated_tokens.append(next_token.item())
    return next_token.item(), attention

def top_k_sampling(logits,top_p=0.8,filter_value=-float('inf')):
    assert logits.dim() == 1
    sorted_logits, sorted_indices = torch.sort(logits,descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits,dim=-1),dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
    sorted_indices_to_remove[...,0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def beam_search(logits,k=10):
    pass


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


