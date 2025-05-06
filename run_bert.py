# coding: UTF-8
import time
import torch
import numpy as np
from train_eval_new import train
from importlib import import_module
import argparse
from utils import *

parser = argparse.ArgumentParser(description='Profanity detection')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

dataset = 'COLD'  
module = import_module('models.' + args.model )
config = import_module('models.configs.' + args.model).Config(dataset)
set_seeds(1)
start = time.time()
train_data, dev_data, test_data = build_dataset(config)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
end=time.time()

model = module.Model(config).to(config.device)
train(config, model, train_iter, dev_iter, test_iter)
