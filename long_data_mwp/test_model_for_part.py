import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer
from tqdm import tqdm
import multiprocessing
import pickle
import numpy as np

token_map = {'true':2995, 'false':6270, '≈':1606, '≡':1607}

seed = 42
device = torch.device("cuda:0")

# training parameters
batch_size = 8
max_epoch = 100               # number of training epoch
learning_rate = 0.00005        # learning rate
early_stopping_step = 10

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

def train(tr_set, dv_set, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            print('\n-----epoch= {:3d}, dev_loss= [{:.5f}], dev_acc= [{:.5f}]'.format(epoch + 1, dev_loss, dev_acc), end=', ')
            print('\n\t train loss=[{:.5f}], train acc=[{:.5f}]'.format(record['train']['loss'][-1], record['train']['acc'][-1]), end=', ')
            torch.save(model.state_dict(), 'models/fine_mwp_multi_equal_lr_00005.pth')
        else:
            early_stopping_count += 1
        epoch += 1
    if early_stopping_count>=early_stopping_step:
        print('Early stopping !!!')
    else:
        print('Finished training !!!')
    return epoch + 1, record

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

def test_for_part(tt_set, model):
    model.eval()
    all_acc = [1] * test_sample_num
    acc = 0
    preds = []
    for j, (x, label) in enumerate(tqdm(tt_set)):                       
        with torch.no_grad():                   
            pred = model(x, label)
            current_index = j % test_sample_num
            for p in range(len(x)):
                mask_index = model.tokenizer(x[p])['input_ids'].index(model.tokenizer.mask_token_id)
                preds.append(torch.argmax(pred['logits'][p, mask_index]).detach().cpu().item())
                if torch.argmax(pred['logits'][p, mask_index]).detach().cpu().item() == model.tokenizer(label[p])['input_ids'][mask_index]:
                    all_acc[current_index] = all_acc[current_index] * 1
                    acc += 1
                else:
                    # print(x)
                    all_acc[current_index] = 0
    test_acc = sum(all_acc) / test_sample_num
    test_acc_for_part = acc / len(tt_set.dataset)
    return test_acc, test_acc_for_part, preds

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

    multiprocessing.Pool(processes=4)

    # data_name = 'tree_r10_f10_float3'

    # test_data, test_label = read_data('dataset/' + data_name + '/' + data_name + '_test')
    test_data, test_label = read_data('dataset/part_same_feature_german_tree_md10_float3')
   
    tokenizer = AutoTokenizer.from_pretrained("../mwpbert/MWP-BERT_en")
    model = BertForMaskedLM.from_pretrained("../mwpbert/MWP-BERT_en")

    LM = MaskLM(model, tokenizer, device)

    tt_set = prep_dataloader('test', test_data, test_label, 1)
    test_sample_num = 1000 # german
    # test_sample_num = 10000 # fake german

    r_f = 10

    LM.load_state_dict(torch.load(f'models/pad_tree_r{r_f}_f{r_f}_float3.pth'))
    # LM.load_state_dict(torch.load(f'models/random_feature_tree_r{r_f}_f{r_f}_float3.pth'))

    test_acc, test_acc_for_part, preds = test_for_part(tt_set, LM)
    print(f'acc = {test_acc}')
    print(f'acc for part = {test_acc_for_part:.3f}')
