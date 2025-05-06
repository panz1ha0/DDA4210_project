# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        _, pooled = self.bert(x[0], attention_mask=x[2], output_all_encoded_layers=False)
        out = self.linear(pooled)
        return out
