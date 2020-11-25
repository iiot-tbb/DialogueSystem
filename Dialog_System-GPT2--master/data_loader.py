import torch
import json
import random
import math
from torch.utils.data import DataLoader,Dataset,Sampler
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2Tokenizer

class BucketSampler(Sampler):
    """Sort data by sequence length"""
    def __init__(self, lens, bucket_size, batch_size,droplast=False,
                 shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda  i: self._lens[i],reverse=True)
                   for i in range(0, len(ids),self._bucket_size)]

        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0,len(bucket),self._batch_size)]

        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size] * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])

        if self._droplast:
            return sum(s // self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)



class GPT2FeatureDataset(Dataset):

    def __init__(self, features, max_len = None):
        self.features = features
        self.max_len = max_len

    def __getitem__(self, i):
        feat_dict = self.features[i]
        if self.max_len is not None and feat_dict['input_len'] > self.max_len:
            # truncate on the left side
            feat_dict['input_ids'] = feat_dict['input_ids'][-self.max_len:]
            feat_dict['position_ids'] = feat_dict['position_ids'][-self.max_len:]
            feat_dict['token_type_ids'] = feat_dict['token_type_ids'][-self.max_len:]
            feat_dict['lm_labels'] = feat_dict['lm_labels'][-self.max_len:]

        feat = InputFeatures(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        inputs_ids = pad_sequence([torch.tensor(f.input_ids,dtype=torch.long)
                                   for f in features],
                                  batch_first=True,padding_value=0)
        position_ids = pad_sequence([torch.tensor(f.position_ids,dtype = torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids, dtype = torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        return (inputs_ids, position_ids, token_type_ids, labels)

class InputFeatures(object):
    def __init__(self,input_ids, position_ids, token_type_ids,
                 lm_labels=None, weights=None, input_len = None):
        #self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.lm_labels = lm_labels
        self.weights = weights
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


class GPT2DataLoader(object):

    def __init__(self,data_path, vocab_file, bpe_merges,bucket,batch_size, max_seq_len,
                 shuffle=True,end_text = "<|endoftext|>"):

        self.vocab_file = vocab_file
        self.bpe_merges = bpe_merges
        self.end_text = end_text
        self.data_path = data_path
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_len = max_seq_len

    def __iter__(self):
        dials = self.read_data(self.data_path)
        features = self.build_input_features(dials,self.vocab_file,self.bpe_merges)
        trunc_chunk = []
        lens = []
        for feat in features:
            if feat['input_len'] > self.max_len:
                continue
            trunc_chunk.append(feat)
            lens.append(feat['input_len'])
        dataset = GPT2FeatureDataset(trunc_chunk,self.max_len)

        sampler = BucketSampler(lens,self.bucket_size, self.batch_size,
                                droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0,collate_fn=GPT2FeatureDataset.collate)
        yield from loader

    def __len__(self):
        return len(self.read_data(self.data_path))//self.batch_size

    @staticmethod
    def read_data(data_path):
        with open(data_path, 'r') as data:
            lines = [l.strip() for l in data]
        dials = []
        for l in lines:
            if l == "<dial>":
                dial = []
            elif l == '<\dial>':
                dials += [dial]
            else:
                dial += [l.strip()]
        return dials

    @staticmethod
    def build_input_features(dials, vocab_file, bpe_merges, end_text="<|endoftext|>"):
        tokenizer = GPT2Tokenizer(vocab_file, bpe_merges)
        feature = []
        for dial in dials:
            inputs = sum([tokenizer.encode(u) for u in dial[:-1]], [])
            lm_labels = [-1] * len(inputs) + tokenizer.encode(dial[-1] + end_text)
            token_type_ids = [0] * len(inputs) + [1] * (len(tokenizer.encode(dial[-1] + end_text)))
            weights = [0.0] * len(inputs) + [1.0] * (len(tokenizer.encode(dial[-1] + end_text)))
            input_ids = inputs + tokenizer.encode(end_text + dial[-1])
            input_len = len(input_ids)
            position_ids = list(range(len(input_ids)))

            feat_dict = {"input_ids": input_ids,
                         "position_ids": position_ids,
                         "token_type_ids": token_type_ids,
                         "lm_labels": lm_labels,
                         "weights": weights,
                         "input_len": input_len}
            feature.append(feat_dict)
        return feature


if __name__ == '__main__':
    print("cool")
    data_loader = GPT2DataLoader(data_path='Data/Ubuntu/ub_train.txt',
                                 vocab_file='configs/345M/vocab.json',
                                 bpe_merges='configs/345M/merges.txt',
                                 bucket =2,
                                 batch_size=10,
                                 max_seq_len=1024)

    for x in data_loader:
        print("hello")
        input_ids, position_ids, token_ids, label_ids, *_ = x
        print(input_ids.shape)
        print(input_ids)
        print(label_ids)
       # break


    #dials = read_data('DailyDialog/train_text.txt')
    #f = build_input_features(dials,'./vocab_file/encoder.json',
    #                     'vocab_file/merges.txt')
   # tokenizer = GPT2Tokenizer('./vocab_file/encoder.json',
   #                            'vocab_file/merges.txt')
   # l = torch.tensor([10919, 389, 345, 1804, 329, 257,2877, 5633, 50256, 72, 716, 257, 13169, 764, 0, 0])
   # print(tokenizer.decode(l.numpy()))


