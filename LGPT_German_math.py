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

    German_Dataset = minyu.German('cpu')
    X, y, German_Dataset_features = German_Dataset.preprocess()

    df = pd.DataFrame(
        German_Dataset.x, 
        columns=German_Dataset_features
        )
    
    df['target'] = y

    X_pre, X_fine, y_pre, y_fine = train_test_split(X, y, test_size=0.2, random_state=torch_seed, stratify=y)

    Tree_model = DecisionTreeClassifier(max_depth=3, random_state=torch_seed)
    Tree_model.fit(X_pre, y_pre)

    # Show rules
    rule_text = tree.export_text(Tree_model, feature_names=German_Dataset_features)
    print(rule_text)

    rule_text = '''
    The person\'s credit is bad if (Status of existing checking account less than or equal to 1.50 AND Duration less than or equal to 22.50 AND Credit history less than or equal to 1.50) OR (Status of existing checking account less than or equal to 1.50 AND Duration bigger than 22.50 AND Savings account/bonds less than or equal to 2.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans less than or equal to 0.50 AND Credit amount bigger than 3741.00).
    The person\'s credit is good if (Status of existing checking account less than or equal to 1.50 AND Duration less than or equal to 22.50 AND Credit history bigger than 1.50) OR (Status of existing checking account less than or equal to 1.50 AND Duration bigger than 22.50 AND Savings account/bonds bigger than 2.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans bigger than 0.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans less than or equal to 0.50 AND Credit amount less than or equal to  3741.00).
    '''

    print('Decision Tree ACC')
    print('pretrain')
    Decision_train_acc = Tree_model.score(X_pre, y_pre)
    print(Decision_train_acc)
    print('finetune')
    Decision_tree_acc = Tree_model.score(X_fine, y_fine)
    print(Decision_tree_acc)
    print('Decision Tree MCC')
    print('pretrain')
    print(matthews_corrcoef(Tree_model.predict(X_pre), y_pre))
    print('finetune')
    print(matthews_corrcoef(Tree_model.predict(X_fine), y_fine))

    tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT')
    bertmodel = BertForMaskedLM.from_pretrained('tbs17/MathBERT')
    bertmodel = bertmodel.to(device)
    for param in bertmodel.parameters():
        param.requires_grad = False

    # class_token = 'good bad'  # 2204, 2919
    # class_token = tokenizer(class_token, return_tensors="pt")
    # print(class_token)

    rule_text_token = tokenizer(rule_text, return_tensors="pt")
    rule_text_emb = bertmodel.bert.embeddings(input_ids=rule_text_token['input_ids'], token_type_ids=rule_text_token['token_type_ids'])
    print(rule_text_emb.shape)

    class LLMTree(nn.Module):
        def __init__(self, LLM_model, LLM_tokenizer, batch_size, device):
            super(LLMTree, self).__init__()
            self.LLM_model = LLM_model
            self.LLM_emb_layer = LLM_model.bert.embeddings
            self.tokenizer = LLM_tokenizer
            self.LLM_hidden_dim = 768
            self.class_num = 3
            self.batch_size = batch_size
            self.feature_names = German_Dataset_features
            self.rule_emb = rule_text_emb
            
        def rule_of_sample(self, X):
            batch_sample = []
            for n in range(len(X)):
                sample = 'The person\'s credit is [MASK] because '
                for i in range(len(X[n])):
                    sample += f'{self.feature_names[i]} is {X[n][i]}, '
                sample = sample[:-2] + '.'
                batch_sample.append(sample)
            return batch_sample
            
        def forward(self, x, device):
            x = self.tokenizer(self.rule_of_sample(x), return_tensors="pt")
            x = self.LLM_emb_layer(input_ids=x['input_ids'].to(device), token_type_ids=x['token_type_ids'].to(device))
            x = torch.cat((self.rule_emb.repeat(x.shape[0], 1, 1).to(device), x), 1)  # batch*(rule+x)*768
            x = self.LLM_model(inputs_embeds=x)['logits'] 
            return x
        
    model = LLMTree(bertmodel, tokenizer, 1, device).to(device)

    train_acc = 0
    for i, x in enumerate(tqdm(X_pre)):
        x = torch.tensor(x).unsqueeze(dim=0).to(device)
        pred = model(x, device)
        if pred[0, 258, 2204] > pred[0, 258, 2919]:
            pred = 0
        else:
            pred = 1
        if pred == y_pre[i]:
            train_acc += 1
    print(f'pretrain acc : {train_acc / len(X_pre)}')

    test_acc = 0
    for i, x in enumerate(tqdm(X_fine)):
        x = torch.tensor(x).unsqueeze(dim=0).to(device)
        pred = model(x, device)
        if pred[0, 258, 2204] > pred[0, 258, 2919]:
            pred = 0
        else:
            pred = 1
        if pred == y_fine[i]:
            test_acc += 1
    print(f'finetune acc : {test_acc / len(X_fine)}')