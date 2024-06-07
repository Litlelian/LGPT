import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer
from tqdm import tqdm
import multiprocessing
import pickle
import numpy as np
import sys

token_map = {'true':2995, 'false':6270, '≈':1606, '≡':1607}

# data_name = 'fsw_and_rsw_and_r3_f5_float3'

seed = 42
device = torch.device("cuda:0")

# training parameters
batch_size = 16
max_epoch = 25               # number of training epoch
learning_rate = 0.00001        # learning rate
early_stopping_step = 3

def set_seed(seed=42):
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  

set_seed(seed)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def read_data(filename):
    datas = []
    labels = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            data = str(lines[i+1].strip())
            label = str(lines[i].strip())
            datas.append(data)
            labels.append(label)
    return datas, labels

class compare_dataset(Dataset):
    def __init__(self, data, label, mode='train'):
        self.data = data
        self.target = label
        self.mode = mode
            
    def __getitem__(self, index):
        # Returns one sample at a time
        return self.data[index], self.target[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
    
def prep_dataloader(mode, data, label, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = compare_dataset(data, label, mode)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                 
    return dataloader

def train(tr_set, dv_set, tt_set, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch = 0
    min_loss = 100000.
    record = {'train': {'loss':[], 'acc':[]}, 'dev': {'loss':[], 'acc':[]}}
    early_stopping_count = 0
    while (epoch < max_epoch) and (early_stopping_count<early_stopping_step):
        acc = 0
        print('\n-----epoch= {:3d}'.format(epoch + 1), end='. ')
        model.train()
        loss = 0
        train_loss = 0
        for x, label in tqdm(tr_set):
            optimizer.zero_grad()
            pred = model(x, label)
            loss = pred['loss']
            train_loss += loss
            for p in range(len(x)):
                mask_index = model.tokenizer(x[p])['input_ids'].index(model.tokenizer.mask_token_id)
                if torch.argmax(pred['logits'][p, mask_index]).detach().cpu().item() == model.tokenizer(label[p])['input_ids'][mask_index]:
                    acc += 1
            loss.backward()                
            optimizer.step()                    
        record['train']['loss'].append((train_loss / len(tr_set.dataset)).detach().cpu().item())
        record['train']['acc'].append(acc / len(tr_set.dataset))
        dev_loss, dev_acc = dev(dv_set, model)
        record['dev']['loss'].append(dev_loss)
        record['dev']['acc'].append(dev_acc)
        if dev_loss < min_loss:
            early_stopping_count = 0
            min_loss = dev_loss
            print(f'-----epoch= {epoch + 1}, dev_loss= [{dev_loss:.5f}], dev_acc= [{dev_acc}]')
            print(f'train loss=[{record["train"]["loss"][-1]:.5f}], train acc=[{record["train"]["acc"][-1]}]')
            print(f'test acc = [{test(tt_set, model)}]')
            torch.save(model.state_dict(), 'models/' + data_name + '_' + pretrain_model + '.pth')
        else:
            early_stopping_count += 1
        epoch += 1
    if early_stopping_count>=early_stopping_step:
        print('Early stopping !!!')
        print(epoch - early_stopping_step)
    else:
        print('Finished training !!!')
        print(epoch)
    return epoch , record

def dev(dv_set, model):
    model.eval()
    total_loss = 0
    acc = 0
    for x, label in tqdm(dv_set):                       
        with torch.no_grad():                   
            pred = model(x, label)
            loss = pred['loss']
            for p in range(len(x)):
                mask_index = model.tokenizer(x[p])['input_ids'].index(model.tokenizer.mask_token_id)
                if torch.argmax(pred['logits'][p, mask_index]).detach().cpu().item() == model.tokenizer(label[p])['input_ids'][mask_index]:
                    acc += 1
        total_loss += loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    dev_acc = acc / len(dv_set.dataset)
    return total_loss, dev_acc

def test(tt_set, model):
    model.eval()
    acc = 0
    for x, label in tqdm(tt_set):                       
        with torch.no_grad():                   
            pred = model(x, label)
            for p in range(len(x)):
                mask_index = model.tokenizer(x[p])['input_ids'].index(model.tokenizer.mask_token_id)
                if torch.argmax(pred['logits'][p, mask_index]).detach().cpu().item() == model.tokenizer(label[p])['input_ids'][mask_index]:
                    acc += 1
    test_acc = acc / len(tt_set.dataset)
    return test_acc

class MaskLM(nn.Module):
    def __init__(self, LLM_model, LLM_tokenizer, device):
        super(MaskLM, self).__init__()
        self.LLM_model = LLM_model.to(device)
        self.tokenizer = LLM_tokenizer
        self.device = device
        
    def forward(self, x, y):
        x = self.tokenizer(x, return_tensors="pt", padding=True)
        y = self.tokenizer(y, return_tensors="pt", padding=True)["input_ids"]
        y = torch.where(x.input_ids == self.tokenizer.mask_token_id, y, -100)
        x = self.LLM_model(input_ids=x['input_ids'].to(self.device), token_type_ids=x['token_type_ids'].to(self.device), attention_mask=x['attention_mask'].to(self.device), labels=y.to(self.device))
        return x

if __name__ == "__main__":

    data_name = sys.argv[1]
    pretrain_model = sys.argv[2]

    multiprocessing.Pool(processes=4)

    # train_data, train_label = read_data('dataset/' + data_name + '/' + data_name + '_dev')
    # test_data, test_label = read_data('dataset/' + data_name + '/' + data_name + '_test')

    train_data, train_label = read_data('dataset/' + data_name)
    test_data, test_label = read_data('dataset/' + data_name[5:])

    train_data, dev_data, train_label, dev_label = train_test_split(train_data, train_label, test_size = 0.2, random_state = seed)

    tokenizer = AutoTokenizer.from_pretrained("../mwpbert/MWP-BERT_en")
    model = BertForMaskedLM.from_pretrained("../mwpbert/MWP-BERT_en")

    LM = MaskLM(model, tokenizer, device)

    LM.load_state_dict(torch.load("models/" + pretrain_model + ".pth"))

    tr_set = prep_dataloader('train', train_data, train_label, batch_size)
    dv_set = prep_dataloader('dev', dev_data, dev_label, batch_size)
    tt_set = prep_dataloader('test', test_data, test_label, 1)   

    model_epoch, model_record = train(tr_set, dv_set, tt_set, LM)

    with open("record/" + data_name + '_' + pretrain_model + ".pkl", "wb") as tf:
        pickle.dump(model_record, tf)

    LM.load_state_dict(torch.load("models/" + data_name + '_' + pretrain_model + ".pth"))

    test_acc = test(tt_set, LM)
    print(f'pretrain : {pretrain_model} / finetune : {data_name}')
    print(test_acc)