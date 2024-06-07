import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
# from transformers import BertTokenizer, BertForMaskedLM
from bert_nli.bert_nli import BertNLIModel
from tqdm import tqdm
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
    Class 1: If (Status of existing checking account less than or equal to 1.50 AND Duration less than or equal to 22.50 AND Credit history less than or equal to 1.50) OR (Status of existing checking account less than or equal to 1.50 AND Duration bigger than 22.50 AND Savings account/bonds less than or equal to 2.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans less than or equal to 0.50 AND Credit amount bigger than 3741.00).
    Class 0: If (Status of existing checking account less than or equal to 1.50 AND Duration less than or equal to 22.50 AND Credit history bigger than 1.50) OR (Status of existing checking account less than or equal to 1.50 AND Duration bigger than 22.50 AND Savings account/bonds bigger than 2.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans bigger than 0.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans less than or equal to 0.50 AND Credit amount less than or equal to  3741.00).
    '''

    rule_text_c1 = '''
    If (Status of existing checking account less than or equal to 1.50 AND Duration less than or equal to 22.50 AND Credit history less than or equal to 1.50) OR (Status of existing checking account less than or equal to 1.50 AND Duration bigger than 22.50 AND Savings account/bonds less than or equal to 2.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans less than or equal to 0.50 AND Credit amount bigger than 3741.00).
    '''

    rule_text_c0 = '''
    If (Status of existing checking account less than or equal to 1.50 AND Duration less than or equal to 22.50 AND Credit history bigger than 1.50) OR (Status of existing checking account less than or equal to 1.50 AND Duration bigger than 22.50 AND Savings account/bonds bigger than 2.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans bigger than 0.50) OR (Status of existing checking account bigger than 1.50 AND Other installment plans less than or equal to 0.50 AND Credit amount less than or equal to  3741.00).
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

    bert_type = 'bert-base'
    bertmodel = BertNLIModel('bert_nli/output/{}.state_dict'.format(bert_type), bert_type=bert_type, gpu=False)
    bertmodel = bertmodel.to(device)
    for param in bertmodel.parameters():
        param.requires_grad = False

    # labels_1, prob_1 = bertmodel([(rule_text_c1,rule_text_c0)])
    # print(labels_1)  # ['entail']
    # print(prob_1)  # [[0.0325726  0.9441019  0.02332539]]

    for i, x in enumerate(tqdm(X_pre)):
        x_sample = ''
        for n in range(len(x)):
            x_sample += f'{German_Dataset_features[n]} is {x[n]}, '
        x_sample = x_sample[:-2] + '.'
        labels_1, prob_1 = bertmodel([(rule_text_c1,x_sample)])
        labels_0, prob_0 = bertmodel([(rule_text_c0,x_sample)])
        print(y_pre[i])
        print(prob_1)
        print(prob_0)
        if i == 3:
            print(ujfjfpofn)