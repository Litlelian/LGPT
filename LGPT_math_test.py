import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import train_test_split
from transformers import *
from tqdm import tqdm
import pandas as pd
import data_preprocess as minyu
from sklearn.metrics import matthews_corrcoef
import multiprocessing

torch_seed = 42 
torch.manual_seed(torch_seed)
torch.cuda.manual_seed_all(torch_seed)
torch.cuda.manual_seed(torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    multiprocessing.Pool(processes=4)

    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT-custom')
    bertmodel = BertForPreTraining.from_pretrained('tbs17/MathBERT-custom')
    # tokenizer = AutoTokenizer.from_pretrained('mathbert_downstream')
    # bertmodel = BertForPreTraining.from_pretrained('mathbert_downstream', from_tf=True)

    bertmodel = bertmodel.to(device)
    for param in bertmodel.parameters():
        param.requires_grad = False

    # class_token = 'good bad'  # 2204, 2919
    # class_token = 'true false'  # 2995, 6270
    # class_token = tokenizer(class_token, return_tensors="pt")
    # print(class_token)
        
    rule_text = '200 > 180, which is true. 100 < 80, which is false.'

    rule_text_token = tokenizer(rule_text, return_tensors="pt")
    rule_text_emb = bertmodel.embeddings(input_ids=rule_text_token['input_ids'], token_type_ids=rule_text_token['token_type_ids'])
    print(rule_text_emb.shape)

    class LLMTree(nn.Module):
        def __init__(self, LLM_model, LLM_tokenizer, batch_size, device):
            super(LLMTree, self).__init__()
            self.LLM_model = LLM_model
            self.LLM_emb_layer = LLM_model.embeddings
            self.tokenizer = LLM_tokenizer
            self.LLM_hidden_dim = 768
            self.class_num = 3
            self.batch_size = batch_size
            self.rule_emb = rule_text_emb
            
        def forward(self, x, device):
            x = self.tokenizer(x, return_tensors="pt")
            x = self.LLM_emb_layer(input_ids=x['input_ids'].to(device), token_type_ids=x['token_type_ids'].to(device))
            x = torch.cat((self.rule_emb.repeat(x.shape[0], 1, 1).to(device), x), 1)  # batch*(rule+x)*768
            x = self.LLM_model(inputs_embeds=x)['seq_relationship_logits'] 
            return x
        
    model = LLMTree(bertmodel, tokenizer, 1, device).to(device)
    # x = 'The person is [MASK] if his IQ < 180.'
    x = '200 > 180, which is [MASK].'

    pred = model(x, device)
    print(pred)