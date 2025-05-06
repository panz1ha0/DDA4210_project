# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_size)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_size), config.num_classes)


    def forward(self, x): 
        
        encoder_out, _ = self.bert(x[0], attention_mask=x[2], output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        conv_and_pool = lambda x, conv: F.max_pool1d(F.relu(conv(x)).squeeze(3), F.relu(conv(x)).squeeze(3).size(2)).squeeze(2)
        out = torch.cat([conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out

