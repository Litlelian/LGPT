import random
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import datasets
import data_preprocess as minyu

def min_max_norm(data):
    norm_data = np.copy(data)
    max_data = np.max(data, axis=0)
    min_data = np.min(data, axis=0)
    for i, d in enumerate(data):
        for f in range(len(d)):
            norm_data[i, f] = 2 * ((d[f] - min_data[f]) / (max_data[f] - min_data[f])) - 1
    return np.around(norm_data, 3)

def get_all_tree_paths(tree):
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
            paths[node] = path
    
    recurse(0, [])
    return paths

def train_DT(data, target):
    Tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_seed)
    Tree_model.fit(data, target)
    rule_t = []
    rule_f = []
    all_tree_paths = get_all_tree_paths(Tree_model.tree_)
    all_tree_leaves = list(all_tree_paths.keys())
    path_count = {}
    for tl in all_tree_leaves:
        path_count[tl] = 0
    for i in range(data.shape[0] // 2):  # true
        leaf_id = Tree_model.apply(data[i].reshape(1, -1))[0]
        path_count[leaf_id] += 1
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "≡ "
        for rt in all_tree_paths[leaf_id]:
            rule += rt
            rule += " "
        rule_t.append(rule)
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "[MASK] "
        for rt in all_tree_paths[leaf_id]:  # should pad here
            rule += rt
            rule += " "
        rule_t.append(rule)
    for i in range(data.shape[0] // 2, data.shape[0]):  # false
        leaf_id = Tree_model.apply(data[i].reshape(1, -1))[0]
        # select a wrong path
        other_paths = [x for x in all_tree_leaves if x != leaf_id]
        random_leaf_id = random.choice(other_paths)
        path_count[random_leaf_id] += 1
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "≈ "
        for rt in all_tree_paths[random_leaf_id]:
            rule += rt
            rule += " "
        rule_f.append(rule)
        rule = ""
        for n in range(feature_num):
            rule += f"(f{n + 1} = {data[i, n]:.3f}) "
        rule += "[MASK] "
        for rt in all_tree_paths[random_leaf_id]:  # should pad here
            rule += rt
            rule += " "
        rule_f.append(rule)
    return rule_t, rule_f, path_count

def generate_data(data_name, data, target):
    all_rule = []
    rule_t, rule_f, path_count = train_DT(data, target)
    print(path_count)
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
max_depth = 20
data_name = f'german_tree_md{max_depth}_float{3}'

generate_data(data_name, real_x, real_y)