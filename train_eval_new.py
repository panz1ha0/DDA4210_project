# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
from tqdm import tqdm
import matplotlib.pyplot as plt


def init_network(model, seed=0):
    for name, w in model.named_parameters():
        if len(w.size()) < 2:
            continue
        if 'weight' in name:
            nn.init.xavier_normal_(w)
        elif 'bias' in name:
            nn.init.constant_(w, 0)


def train(config, model, train_iter, dev_iter, test_iter, name='bert'):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0  
    train_loss = []
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            train_loss.append(loss.tolist())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                pred = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, pred)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                model.train()
            total_batch += 1
    torch.save(model.state_dict(), config.save_path)
    y_train_loss = train_loss
    x_train_loss = range(len(y_train_loss))

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('loss')
    
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.show()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print(test_report)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        return acc, loss_total / len(data_iter), report
    return acc, loss_total / len(data_iter)


def predict(config, model, device, pad_size=32):
    with open("./predict.txt", 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        f.close
    contents = []
    PAD, CLS = '[PAD]', '[CLS]'
    for line in lines:
        token = config.tokenizer.tokenize(line)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, seq_len, mask))
    with torch.no_grad():
        with open("./predict_out.txt", 'w', encoding='UTF-8') as f:
            input1 = torch.LongTensor([_[0] for _ in contents])
            input2 = torch.LongTensor([_[1] for _ in contents])
            input3 = torch.LongTensor([_[2] for _ in contents])
            input = (input1, input2, input3)
            # print(type(contents))
            out = model(input)
            _, pre = torch.max(out.data, 1)
            for item in pre.tolist():
                f.write(str(item))
                f.write("\n")
        f.close()