import random
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import datasets
import data_preprocess as minyu

def cutting_feature(all_feature_num, slice_size):
    cf_list = []
    for i in range(all_feature_num // slice_size):
        cf_list.append([i * slice_size, (i + 1) * slice_size])
    if all_feature_num % slice_size != 0:
        cf_list.append([(i + 1) * slice_size, all_feature_num])

    return cf_list

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

def pad_paths(paths, feature_part):
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

def min_max_norm(data):
    norm_data = np.copy(data)
    max_data = np.max(data, axis=0)
    min_data = np.min(data, axis=0)
    for i, d in enumerate(data):
        for f in range(len(d)):
            norm_data[i, f] = 2 * ((d[f] - min_data[f]) / (max_data[f] - min_data[f])) - 1
    return np.around(norm_data, 3)

def get_all_tree_paths(tree, feature_part):
    # feature_part = [start, end]
    paths = {}
    features_name = []
    
    for i in range(feature_num):
        features_name.append(f'f{i + 1}')

    def recurse(node, path):
        if tree.feature[node] != -2:  # -2 is leaf node
            name = features_name[tree.feature[node]]
            threshold = tree.threshold[node]
            if int(name[1:]) > feature_part[0] and int(name[1:]) <= feature_part[1]:
                left_decision = f"(f{int(name[1:]) - feature_part[0]} <= {threshold:.3f})"
                right_decision = f"(f{int(name[1:]) - feature_part[0]} > {threshold:.3f})"
                recurse(tree.children_left[node], path + [left_decision])
                recurse(tree.children_right[node], path + [right_decision])
            else:
                recurse(tree.children_left[node], path)
                recurse(tree.children_right[node], path)
        else:
            paths[node] = [path]
    
    recurse(0, [])

    for p in paths.keys():
        rule_of_each_f = [0] * (feature_part[1] - feature_part[0])
        for f in paths[p][0]:
            ff = f.split('f')[1].split(' ')[0]
            rule_of_each_f[int(ff) - 1] += 1
        paths[p].append(rule_of_each_f)

    no_repeat_paths, alias_paths = remove_duplicate_list_values(paths)

    padding_paths = pad_paths(no_repeat_paths, feature_part)

    balance_paths = {}
    
    for key, value in padding_paths.items():
        balance_paths[key] = value[0]

    return balance_paths, alias_paths

def train_DT(data, target, feature_part):
    Tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_seed)
    Tree_model.fit(data, target)
    rule_t = []
    rule_f = []
    all_tree_paths, alias_paths = get_all_tree_paths(Tree_model.tree_, feature_part)
    all_tree_leaves = list(all_tree_paths.keys())
    for i in range(data.shape[0]):  # true
        leaf_id = Tree_model.apply(data[i].reshape(1, -1))[0]
        alias_leaf_id = alias_paths[leaf_id]
        rule = ""
        for n in range(feature_part[0], feature_part[1]):
            rule += f"(f{n + 1 - feature_part[0]} = {data[i, n]:.3f}) "
        rule += "≡ "
        for rt in all_tree_paths[alias_leaf_id]:
            rule += rt
            rule += " "
        rule_t.append(rule)
        rule = ""
        for n in range(feature_part[0], feature_part[1]):
            rule += f"(f{n + 1 - feature_part[0]} = {data[i, n]:.3f}) "
        rule += "[MASK] "
        for rt in all_tree_paths[alias_leaf_id]:  # should pad here
            rule += rt
            rule += " "
        rule_t.append(rule)
    # for i in range(data.shape[0] // 2, data.shape[0]):  # false
    #     leaf_id = Tree_model.apply(data[i].reshape(1, -1))[0]
    #     # select a wrong path
    #     other_paths = [x for x in all_tree_leaves if x != leaf_id]
    #     if len(other_paths) == 0:
    #         break
    #     random_leaf_id = random.choice(other_paths)
    #     alias_leaf_id = alias_paths[random_leaf_id]
    #     rule = ""
    #     for n in range(feature_part[0], feature_part[1]):
    #         rule += f"(f{n + 1} = {data[i, n]:.3f}) "
    #     rule += "≈ "
    #     for rt in all_tree_paths[alias_leaf_id]:
    #         rule += rt
    #         rule += " "
    #     rule_f.append(rule)
    #     rule = ""
    #     for n in range(feature_part[0], feature_part[1]):
    #         rule += f"(f{n + 1} = {data[i, n]:.3f}) "
    #     rule += "[MASK] "
    #     for rt in all_tree_paths[alias_leaf_id]:  # should pad here
    #         rule += rt
    #         rule += " "
    #     rule_f.append(rule)
    return rule_t, rule_f

def generate_data(data_name, data, target):
    all_rule = []
    for feature_part in cutting_feature(feature_num, max_depth):
        rule_t, rule_f = train_DT(data, target, feature_part)
        all_rule.append(rule_t)
        all_rule.append(rule_f)

    with open('dataset/' + data_name, 'w') as file:
        for i in range(len(all_rule)):
            for r in range(len(all_rule[i])):
                file.write(all_rule[i][r])
                file.write('\n')

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# real_data = datasets.load_iris()
# real_data = datasets.load_wine()
German_Dataset = minyu.German('cpu')
X, y, German_Dataset_features = German_Dataset.preprocess()
# real_x = min_max_norm(real_data.data)
# real_y = real_data.target
real_x = min_max_norm(np.array(X))
real_y = np.array(y)
feature_num = real_x.shape[1]
max_depth = 10
data_name = f'part_same_feature_german_tree_md{max_depth}_float{3}'

generate_data(data_name, real_x, real_y)