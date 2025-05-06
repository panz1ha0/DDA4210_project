import torch
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding='random'):
        self.model_name = 'TextCNN'
        # self.train_path = dataset + '/data/cold_train.txt'
        self.train_path = dataset + '/data/cold_train_combine.txt'                   
        # self.dev_path = dataset + '/data/cold_dev.txt'
        self.dev_path = dataset + '/data/cold_dev_combine.txt'                    
        self.test_path = dataset + '/data/cold_test.txt'
        # self.test_path = dataset + '/data/cold_test_combine.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/new_class.txt', encoding='utf-8').readlines()] 
        self.vocab_path = dataset + '/data/vocab.pkl'                             
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'    
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.dropout = 0.1                                    
        self.require_improvement = 1000                         
        self.num_classes = len(self.class_list)                       
        self.n_vocab = 0                                          
        self.num_epochs = 4                                
        self.batch_size = 64                                           
        self.pad_size = 32                                          
        self.learning_rate = 1e-3                                     
        self.embed = 300
        self.filter_sizes = (2, 3, 4)                                  
        self.num_filters = 512                                        