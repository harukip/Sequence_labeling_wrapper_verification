import numpy as np
import os
from tqdm import tqdm

def max_num_set(set_total, set_data_count):
    max_set = []
    for i in range(set_total):
        max_set.append(0)
    for sets in range(len(set_data_count)):
        max_set[sets] = max(set_data_count[sets])
    return max_set

def feature_padding_set(df, set_count, set_num, max_set):
    feature = []
    count = 0
    for pages in set_count[set_num]:
        set_len = pages
        for i in range(set_len):
            feature.append(df[count])
            count += 1
        if set_len != max_set[set_num]:
            for i in range(max_set[set_num] - set_len):
                feature.append(9999)
    return feature

def one_of_n(ans, total):
    tmp = []
    for i in range(int(total)):
        if ans == i:
            tmp.append(1.0)
        else:
            tmp.append(0.0)
    return tmp

def label_padding_set(df, set_count, set_num, max_set):
    label = []
    count = 0
    for pages in set_count[set_num]:
        set_len = pages
        for i in range(set_len):
            label.append(df[count])
            count += 1
        if set_len != max_set[set_num]:
            for i in range(max_set[set_num] - set_len):
                label.append(0)
    return label

def CRFSuite_process_set(df, set_count, set_num, max_set):
    cols = ['Leafnode', 'PTypeSet', 'TypeSet', 'Contentid', 'Pathid', 'Simseqid']
    features = []
    for c in range(len(cols)):
        features.append(np.array(feature_padding_set(df[cols[c]], set_count, set_num, max_set)))
        features[c] = np.expand_dims(features[c], -1)
    
    features = np.concatenate([feature for feature in features], -1)
    features = np.reshape(features, [len(set_count[set_num]) * max_set[set_num], 6])
    
    label = np.array(label_padding_set(df['Label'], set_count, set_num, max_set))
    label = np.reshape(label, [len(set_count[set_num]) * max_set[set_num]])
    return features, label

def get_col_type(current_path, set_num):
    col_type = []
    with open(os.path.join(current_path, "set", "Set-"+str(set_num + 1)+"_coltype.txt"), "r") as file:
        tmp = file.readline()
        slot = eval(tmp)
        col_type = slot
    return col_type

def predict_output(current_path, model_name, set_num, col_type, result, max_label, feature_test, max_num):
    Set = []
    with open(os.path.join(current_path, model_name, "set", "set-" + str(set_num + 1)+".csv"), "w") as file:
        for col in col_type: # loop to write the Col type
            file.write(col + "\t")
        file.write("\n")
        current_pos = 1
        for page in tqdm(range(len(result))): # Loop each page
            p_tmp = []
            for cols in range(int(max_label) + 1):
                c_tmp = []
                for node in range(len(result[page])):
                    r = result[page][node]
                    if r == cols:
                        c_tmp.append(node)
                p_tmp.append(c_tmp)
            Set.append(p_tmp)
        Set_tmp = Set.copy()
        for page in range(len(Set_tmp)):
            empty = False
            col = []
            for i in range(len(Set_tmp[page])):
                col.append(False)
            col[0] = True
            while(not empty):
                for cols in range(len(Set_tmp[page])):
                    if len(Set_tmp[page][cols]) == 0:
                        col[cols] = True
                        if cols != 0:
                            file.write("\t")
                    else:
                        n = str(int(feature_test[page * max_num + Set_tmp[page][cols][0]][0]))
                        if cols != 0:
                            file.write(n + "\t")
                        del Set_tmp[page][cols][0]
                        if len(Set_tmp[page][cols]) == 0:
                            col[cols] = True
                    empty = True
                    for i in col:
                        if i == False:
                            empty = False
                            break
                file.write("\n")