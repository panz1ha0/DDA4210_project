# coding: UTF-8
import os
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
import time
import jieba
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'
MAX_VOCAB_SIZE = 10000  
UNK, PAD = '<UNK>', '<PAD>' 

def load_dataset(config, path, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            token = [CLS] + config.tokenizer.tokenize(content)
            seq_len = pad_size
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            token_ids, mask = data_padding(token_ids, pad_size)
            contents.append((token_ids, int(label), seq_len, mask))
    return contents

def load_dataset_others(path, vocab, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            token = jieba_tokenizer(content)
            seq_len = pad_size
            mask = []
            token_ids = []
            for word in token:
                token_ids.append(vocab.get(word, vocab.get(UNK)))
            token_ids, mask = data_padding(token_ids, pad_size)
            contents.append((token_ids, int(label), seq_len, mask))
    return contents

def build_vocab(file_path, tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, _ = lin.split('\t')
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def jieba_tokenizer(content):
    res = jieba.cut(content)
    return list(res)

def data_padding(token_ids, pad_size=32):
    mask = [1] * len(token_ids) + [0] * (pad_size)
    token_ids += [0] * pad_size
    mask = mask[:pad_size]
    token_ids = token_ids[:pad_size]
    return token_ids, mask

def build_dataset_others(config):
    tokenizer = lambda x: jieba_tokenizer(x)
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    train = load_dataset_others(config.train_path, vocab, config.pad_size)
    dev = load_dataset_others(config.dev_path, vocab, config.pad_size)
    test = load_dataset_others(config.test_path, vocab, config.pad_size)
    return vocab, train, dev, test

def build_dataset(config):
    train = load_dataset(config, config.train_path, config.pad_size)
    dev = load_dataset(config, config.dev_path, config.pad_size)
    test = load_dataset(config, config.test_path, config.pad_size)
    return train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = len(batches) % self.n_batches != 0
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        inputs = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        labels = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (inputs, seq_len, mask), labels
    
    def __next__(self):
        batches = self.batches
        start = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        if start >= len(batches):
            self.index = 0
            raise StopIteration
        if end >= len(batches):
            end = len(batches)
        batches = self.batches[start: end]
        self.index += 1
        batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self
 
    def __getitem__(self, item):
        return self.batches
 
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
        
def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  

if __name__ == "__main__":
    train_dir = "./COLD/data/cold_train.txt"
    vocab_dir = "./COLD/data/vocab.pkl"

    tokenizer = lambda x: jieba_tokenizer(x)
    if os.path.exists(vocab_dir):
        vocab = pkl.load(open(vocab_dir, 'rb'))
    else:
        vocab = build_vocab(train_dir, tokenizer=tokenizer)
        pkl.dump(vocab, open(vocab_dir, 'wb'))
    print(f"Vocab size: {len(vocab)}")