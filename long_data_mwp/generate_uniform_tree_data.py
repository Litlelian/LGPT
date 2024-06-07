import random
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def feature_generate(bound, data_len, float_num):
    return np.round(np.random.uniform(-bound, bound, size = (data_len)), float_num)

def remove_duplicate_list_values(input_dict):
    seen_values = set()
    result_dict = {}
    alias_dict = {}
    
    for key, value in input_dict.items():
        value_tuple = tuple(value[0])
        if value_tuple not in seen_values:
            seen_values.add(value_tuple)
            result_dict[key] = value 
            alias_dict[key] = key
        else:
            for o_key, o_value in result_dict.items():
                if value == o_value:
                    alias_dict[key] = o_key
                    break
    
    return result_dict, alias_dict

def pad_paths(paths):
    for key, value in paths.items():
        feature_list = []
        for i, f in enumerate(value[1]):
            if f > 0:
                for x in range(f):
                    feature_list.append(value[0].pop(0))
            else:
                feature_list.append(f'(f{i + 1} <= 1.000)')
        paths[key][0] = feature_list
    return paths

def get_all_tree_paths(tree):
    # feature_part = [start, end]
    paths = {}
    features_name = []
    
    for i in range(feature_num):
        features_name.append(f'f{i + 1}')

    def recurse(node, path):
        if tree.feature[node] != -2:  # -2 is leaf node
            name = features_name[tree.feature[node]]
            threshold = tree.threshold[node]
            left_decision = f"({name} <= {threshold:.3f})"
            right_decision = f"({name} > {threshold:.3f})"
            recurse(tree.children_left[node], path + [left_decision])
            recurse(tree.children_right[node], path + [right_decision])
        else:
            paths[node] = [path]
    
    recurse(0, [])

    for p in paths.keys():
        rule_of_each_f = [0] * feature_num
        for f in paths[p][0]:
            ff = f.split('f')[1].split(' ')[0]
            rule_of_each_f[int(ff) - 1] += 1
        paths[p].append(rule_of_each_f)

    no_repeat_paths, alias_paths = remove_duplicate_list_values(paths)

    padding_paths = pad_paths(no_repeat_paths)

    balance_paths = {}
    
    for key, value in padding_paths.items():
        balance_paths[key] = value[0]

    return balance_paths, alias_paths

def train_DT(data, rule_num):
    # random_feature_bias = random.randint(0, 10) * feature_num
    Tree_model = DecisionTreeClassifier(max_depth=rule_num, random_state=random_seed)
    label = np.random.randint(2, size = data.shape[0])
    Tree_model.fit(data, label)
    rule_t = []
    rule_f = []
    all_tree_paths, alias_paths = get_all_tree_paths(Tree_model.tree_)
    all_tree_leaves = list(all_tree_paths.keys())
    for i in range(data.shape[0] // 2):  # true
        leaf_id = Tree_model.apply(data[i].reshape(1, -1))[0]
        alias_leaf_id = alias_paths[leaf_id]
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "≡ "
        for rt in all_tree_paths[alias_leaf_id]:
            rule += rt
            rule += " "
        rule_t.append(rule)
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "[MASK] "
        for rt in all_tree_paths[alias_leaf_id]:  # should pad here
            rule += rt
            rule += " "
        rule_t.append(rule)
    for i in range(data.shape[0] // 2, data.shape[0]):  # false
        leaf_id = Tree_model.apply(data[i].reshape(1, -1))[0]
        # select a wrong path
        other_paths = [x for x in all_tree_leaves if alias_paths[x] != leaf_id]
        random_leaf_id = random.choice(other_paths)
        alias_leaf_id = alias_paths[random_leaf_id]
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "≈ "
        for rt in all_tree_paths[alias_leaf_id]:
            rule += rt
            rule += " "
        rule_f.append(rule)
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "[MASK] "
        for rt in all_tree_paths[alias_leaf_id]:  # should pad here
            rule += rt
            rule += " "
        rule_f.append(rule)
    return rule_t, rule_f

def generate_data(dataset_length, tree_num, bound, data_folder, data_path):
    all_rule = []
    for t in range(tree_num):
        tree_datas = np.zeros((dataset_length, feature_num))
        for i in range(feature_num):
            tree_datas[:, i] = feature_generate(bound, dataset_length, float_num)
        rule_t, rule_f = train_DT(tree_datas, rule_num)
        all_rule.append(rule_t)
        all_rule.append(rule_f)

    with open('dataset/' + data_folder + '/' + data_path, 'w') as file:
        for i in range(len(all_rule)):
            for r in range(len(all_rule[i])):
                file.write(all_rule[i][r])
                file.write('\n')

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

rule_num = 10
feature_num = 10
float_num = 3
data_name = f'pad_tree_r{rule_num}_f{feature_num}_float{float_num}'

if not os.path.exists('dataset/' + data_name):
    os.mkdir('dataset/' + data_name)

dataset_length = 500
tree_num = 100
bound = 1
data_path = data_name + '_train'
generate_data(dataset_length, tree_num, bound, data_name, data_path)

dataset_length = 100
tree_num = 100
bound = 1
data_path = data_name + '_dev'
generate_data(dataset_length, tree_num, bound, data_name, data_path)

dataset_length = 50
tree_num = 20
bound = 1
data_path = data_name + '_test'
generate_data(dataset_length, tree_num, bound, data_name, data_path)