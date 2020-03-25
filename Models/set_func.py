import numpy as np
import os
import pandas as pd
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



def get_col_type(current_path, set_num):
    col_type = []
    with open(os.path.join(current_path, "set", "Set-"+str(set_num + 1)+"_coltype.txt"), "r") as file:
        tmp = file.readline()
        slot = eval(tmp)
        col_type = slot
    return col_type

def predict_output(current_path, model_name, set_num, col_type, result, max_label, feature_test, max_num):
    Set = []
    with open(os.path.join(current_path, model_name, "set-" + str(set_num + 1)+".csv"), "w") as file:
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

def Set_train_file_generate(set_total, current_path, model_name, data, word_data, max_num):
    set_data_count = []
    df = []
    if set_total > 0:
        for set_t in range(set_total):
            with open(os.path.join(current_path, "data", "Set-"+ str(set_t+1) +".txt"), "r") as set_file:
                set_tmp = []
                output_name = os.path.join(current_path, model_name, "set", "Set-"+ str(set_t+1) +"_train_raw.csv")
                output = open(output_name, "w")
                output.write("Leafnode\tPTypeSet\tTypeSet\tContentid\tPathid\tSimseqid\tPath\tContent\tLabel\n")
                line = set_file.readline()
                slot = line.rstrip("\n").split("\t")
                while(slot[0]!="ColType"): 
                    line = set_file.readline()
                    slot = line.rstrip("\n").split("\t")
                with open(os.path.join(current_path, "set", "Set-"+ str(set_t+1) +"_coltype.txt"), "w") as col_file:
                    col_file.write(str(slot[1:]))
                line = set_file.readline() # First line of data
                page_num = 0
                count = 0
                data_list = [[], [], [], [], [], [], [], [], []]
                while(line != ""):
                    slot = line.rstrip("\n").split("\t")
                    data_info = slot[0].split("-")
                    if(page_num != int(data_info[1])):
                        set_tmp.append(count)
                        count = 0
                    set_num = int(data_info[0])
                    page_num = int(data_info[1])
                    idx = 1
                    sub_list = slot[1:]
                    while("" in sub_list):
                        sub_list.remove("")
                    while(" " in sub_list):
                        sub_list.remove(" ")
                    for element in sub_list:
                        count += 1
                        element = int(element)
                        #print(content_train[page_num][element])
                        num_idx = page_num * max_num + element
                        num_cols = ["Leafnode", "PTypeSet", "TypeSet", "Contentid", "Pathid", "Simseqid"]
                        word_cols = ["Path", "Content"]
                        for c in range(len(num_cols)):
                            output.write(str(data[num_idx][c]) + "\t")
                            data_list[c].append(data[num_idx][c])
                        for c in range(len(word_cols)):
                            if word_data == "":
                                output.write("[]\t")
                                data_list[len(num_cols)+c].append(np.array([]))
                            else:
                                output.write(str(word_data[c][num_idx]) + "\t")
                                data_list[len(num_cols)+c].append(np.array(word_data[c][num_idx]))
                        output.write(str(idx) + "\n")
                        data_list[len(num_cols)+len(word_cols)].append(idx)
                        idx += 1
                    line = set_file.readline()
                set_tmp.append(count)
                output.close()
            set_data_count.append(set_tmp)
            df.append(pd.DataFrame(np.transpose(np.array(data_list)), 
                              columns = ["Leafnode", "PTypeSet", 
                                         "TypeSet", "Contentid", 
                                         "Pathid", "Simseqid", 
                                         "Path", "Content","Label"]))
        with open(os.path.join(current_path, model_name, "set", "set_train_count.txt"), "w") as file:
            file.write(str(set_data_count))
    return df, set_data_count

def Set_test_file_generate(set_total, current_path, model_name, Set_data, data, word_data, max_num):
    set_data_count = []
    df = []
    if set_total > 0:
        for set_t in range(set_total):
            set_tmp = []
            with open(os.path.join(current_path, model_name, "set", "Set-"+ str(set_t+1) +"_ytest_raw.csv"), "w") as set_file:
                set_file.write("Leafnode\tPTypeSet\tTypeSet\tContentid\tPathid\tSimseqid\tPath\tContent\tLabel\n")
                data_list = [[], [], [], [], [], [], [], [], []]
                for pages in tqdm(range(len(Set_data))):
                    count = 0
                    for node in Set_data[pages][set_t]:
                        count += 1
                        node_idx = pages * max_num + int(node)
                        num_cols = ["Leafnode", "PTypeSet", "TypeSet", "Contentid", "Pathid", "Simseqid"]
                        word_cols = ["Path", "Content"]
                        for c in range(len(num_cols)):
                            set_file.write(str(data[node_idx][c]) + "\t")
                            data_list[c].append(data[node_idx][c])
                        for c in range(len(word_cols)):
                            if word_data == "":
                                set_file.write("[]\t")
                                data_list[len(num_cols)+c].append(np.array([]))
                            else:
                                set_file.write(str(word_data[c][node_idx]) + "\t")
                                data_list[len(num_cols)+c].append(np.array(word_data[c][node_idx]))
                        set_file.write(str(0) + "\n")
                        data_list[len(num_cols)+len(word_cols)].append(0)
                    set_tmp.append(count)
            set_data_count.append(set_tmp)
            df.append(pd.DataFrame(np.transpose(np.array(data_list)), 
                              columns = ["Leafnode", "PTypeSet", 
                                         "TypeSet", "Contentid", 
                                         "Pathid", "Simseqid", 
                                         "Path", "Content", "Label"]))
        with open(os.path.join(current_path, model_name, "set", "set_test_count.txt"), "w") as file:
            file.write(str(set_data_count))
    return df, set_data_count