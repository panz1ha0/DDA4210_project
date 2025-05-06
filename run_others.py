# coding: UTF-8
import time
import torch
import numpy as np
from train_eval_others import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset_others, build_iterator, set_seeds

parser = argparse.ArgumentParser(description='Profanity detection')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

dataset = 'COLD'  
module = import_module('models.' + args.model)
config = import_module('models.configs.' + args.model).Config(dataset)
set_seeds(1) 
start = time.time()
vocab, train_data, dev_data, test_data = build_dataset_others(config)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
end=time.time()

config.n_vocab = len(vocab)
model = module.Model(config).to(config.device)
if args.model != 'Transformer':
    init_network(model)
print(model.parameters)
train(config, model, train_iter, dev_iter, test_iter)