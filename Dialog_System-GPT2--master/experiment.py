from transformers import GPT2Tokenizer,GPT2LMHeadModel,GPT2Config
import torch
from test import inference
import torch.nn.functional as F

def reinput(text):
    global conditioned_tokens
    conditioned_tokens = tokenizer.encode(text) + [50256]
    print("\n User input: \n"+text + "\n")


def top_k_filtering(logits,top_p=0.9,filter_value=-float('inf')):
    assert logits.dim() == 1
    sorted_logits, sorted_indices = torch.sort(logits,descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits,dim=-1),dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
    sorted_indices_to_remove[...,0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def recalc():
    global conditioned_tokens
    global generated_tokens
    indexed_tokens = conditioned_tokens + generated_tokens
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    logits = predictions[0, -1, :]
    filtered_logits = top_k_filtering(logits)

    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1)
    generated_tokens.append(next_token.item())

    return next_token.item()

def generate():
    global conditioned_tokens
    global generated_tokens
    while True:
        result = recalc()
        if result == 50256:
            print(tokenizer.decode(generated_tokens[:-1]))
            conditioned_tokens += generated_tokens
            generated_tokens = []
            break

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_file = './configs/345M/vocab.json'
    bpe_merges = './configs/345M/merges.txt'
    checkpoint_path = "checkpoints/medium_fs.pkl"

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    weights = torch.load(checkpoint_path)
    medium_config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)
    model = GPT2LMHeadModel(medium_config)

    weights['lm_head.weight'] = weights['lm_head.decoder.weight']
    weights.pop('lm_head.decoder.weight', None)
    model.load_state_dict(weights)
    model.eval()
    model.to('cuda')

    conditioned_tokens = []
    generated_tokens = []

    reinput("Does money buy happiness?")
    generate()

    while True:
        cmd = input()
        if cmd != "":reinput(cmd)
        generate()