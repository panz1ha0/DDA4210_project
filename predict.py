import time
import torch
import numpy as np
from train_eval import train, init_network, predict
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

dataset = 'THUCNews'
x = import_module('models.bert_CNN')
config = x.Config(dataset)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

model = x.Model(config)
checkpoint = torch.load('./THUCNews/saved_dict/bert_CNN.ckpt')
# print(checkpoint)
model.load_state_dict(checkpoint)
predict(config, model, 'cpu')