import torch
from pytorch_pretrained import BertTokenizer 

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        # self.train_path = dataset + '/data/cold_train.txt'
        self.train_path = dataset + '/data/cold_train_combine.txt'
        # self.dev_path = dataset + '/data/cold_dev.txt'
        self.dev_path = dataset + '/data/cold_dev_combine.txt'
        self.test_path = dataset + '/data/cold_test.txt'
        # self.test_path = dataset + '/data/cold_test_combine.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/new_class.txt').readlines()]               
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'   
        self.vocab_path = dataset + '/data/vocab.pkl'     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        self.require_improvement = 1000                               
        self.num_classes = len(self.class_list)                         
        self.num_epochs = 3                                             
        self.batch_size = 64                                
        self.pad_size = 32                                             
        self.learning_rate = 5e-5                                       
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768