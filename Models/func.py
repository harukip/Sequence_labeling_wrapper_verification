import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import tensorflow as tf

def node_num(data):
    '''
    Generate a list of number of nodes that each page have.
    '''
    count = False
    num_list = []
    data = data.astype('int32')
    for index in range(len(data)):
        if data[index] == 0 and count != False:
            num_list.append(data[index - 1] + 1)
        else:
            count = True
    num_list.append(data[len(data) - 1] + 1)
    count = 0
    index_list = []
    for i in num_list:
        if count == 0:
            index_list.append(i - 1)
            count += 1
        else:
            index_list.append(index_list[count - 1] + i)
            count += 1
    return num_list, index_list

def load_data_num(df, istrain):
    '''
    Get the max num of leafnode and return.
    '''
    num, index = node_num(df['Leafnode'])
    if istrain:
        max_label = df['Label'].astype('int32').max()
        return max(num), max_label
    else:
        return max(num)

def label_padding(data, num, max_num):
    '''
    Padding the data with zero when that page is less than max_num leafnode.
    '''
    output = []
    count = 0
    for page_num in num:
        tmp = []
        page = 0
        if page_num == max_num:
            for i in range(page_num):
                tmp.append(data[count])
                count += 1
            page += 1
        else:
            for i in range(page_num):
                tmp.append(data[count])
                count += 1
            for i in range(max_num - page_num):
                tmp.append(0) # Pad label with 0
            page += 1
        output.append(tmp)
    return output

def node_data(data, num, max_num):
    '''
    Padding the data with zero when that page is less than max_num leafnode.
    '''
    output = []
    count = 0
    for page_num in num:
        tmp = []
        page = 0
        if page_num == max_num:
            for i in range(page_num):
                tmp.append(data[count])
                count += 1
            page += 1
        else:
            for i in range(page_num):
                tmp.append(data[count])
                count += 1
            for i in range(max_num - page_num):
                tmp.append(9999)
            page += 1
        output.append(tmp)
    return output

def node_emb(data, num, pad_len, max_num):
    '''
    Padding the emb with empty when that page is less than max_num leafnode
    '''
    output = []
    count = 0
    tmp2 = []
    for j in range(pad_len):
        tmp2.append('0')
    for page_num in num:
        tmp = []
        page = 0
        if page_num == max_num:
            for i in range(page_num):
                tmp.append(data[count])
                count += 1
            page += 1
        else:
            for i in range(page_num):
                tmp.append(data[count])
                count += 1
            for i in range(max_num - page_num):
                tmp.append(tmp2)
            page += 1
        output.append(tmp)
    return output

def one_of_n(ans, total):
    tmp = []
    for i in range(int(total)):
        if ans == i:
            tmp.append(1.0)
        else:
            tmp.append(0.0)
    return tmp

def word2features(sent, i):
    '''
    Convert each feature into CRFSuite input form.
    '''
    leaf = sent[i][0]
    ptyp = sent[i][1]
    typ = sent[i][2]
    content = sent[i][3]
    pathid = sent[i][4]
    sim = sent[i][5]
    features = {
        'ptyp': str(ptyp),
        'typ': str(typ),
        'content': str(content),
        'pathid': str(pathid),
        'sim': str(sim),
    }
    '''
    If current node is not the first node, then add previous node data as additional feature.
    '''
    if leaf != "0":
        ptyp1 = sent[i-1][1]
        typ1 = sent[i-1][2]
        content1 = sent[i-1][3]
        pathid1 = sent[i-1][4]
        sim1 = sent[i-1][5]
        features.update({
            '-1:ptyp': str(ptyp1),
            '-1:typ': str(typ1),
            '-1:content': str(content1),
            '-1:pathid': str(pathid1),
            '-1:sim': str(sim1),
        })
    else:
        features['BOL'] = True
    if i < len(sent)-1:
        next_leaf = sent[i+1][0]
        if next_leaf != 0:
            ptyp1 = sent[i + 1][1]
            typ1 = sent[i + 1][2]
            content1 = sent[i + 1][3]
            pathid1 = sent[i + 1][4]
            sim1 = sent[i + 1][5]
            features.update({
                '+1:ptyp': str(ptyp1),
                '+1:typ': str(typ1),
                '+1:content': str(content1),
                '+1:pathid': str(pathid1),
                '+1:sim': str(sim1),
            })
        else:
            features['EOL'] = True
    else:
        features['EOL'] = True
    return features

def sent2features(sent):
    '''
    Sent the input features in to list of word2feature.
    '''
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    '''
    Convert the labels to string classes.
    '''
    return [str(sent[i]) for i in range(len(sent))]

def get_col_type(current_path):
    col_type = []
    with open(os.path.join(current_path, "data", "TableA.txt"), "r") as file:
        line = file.readline()
        slot = line.rstrip("\n").split("\t")
        while(slot[0]!="ColType"):
            line = file.readline()
            slot = line.rstrip("\n").split("\t")
        col_type = slot[1:]
    return col_type

def predict_output(set_total, current_path, model_name, col_type, result, max_label, col_set_dict):
    Set_data = []
    with open(os.path.join(current_path, model_name, "data", "predictions.csv"), "w") as file: # Create prediction file
        for col in col_type: # loop to write the Col type
            file.write(col + "\t")
        file.write("\n")
        for page in tqdm(range(len(result))): # Loop each page
            sets = []
            for label in range(int(max_label) + 1): # Loop whole label
                if label == 0:
                    continue
                empty = True
                isset = False
                data = []
                for node in range(len(result[page])):
                    if result[page][node] == label:
                        if empty == False and not isset:
                            file.write(" ")
                        empty = False
                        if label in col_set_dict.keys() and set_total > 0: # That col is a Set
                            isset = True
                            data.append(node)
                        else:
                            file.write(str(node))
                if label in col_set_dict.keys() and set_total > 0: # That col is a Set
                    file.write(str(col_set_dict[label])+"-"+str(page))
                    sets.append(data)
                file.write("\t")
            file.write("\n")
            Set_data.append(sets)
    print("Prediction Output Done!")
    if set_total > 0:
        with open(os.path.join(current_path, model_name, "set", "Set_data.txt"), "w") as set_train_file:
            set_train_file.write(str(Set_data))
    return Set_data

def process_time(current_path, model_name, t, ts, page_c):
    with open(os.path.join(current_path, model_name, "data", "time.txt"),"w") as timef:
        print("\ntrain time:"+str(t))
        timef.write("train:"+str(t)+"\n")
        print("test time:"+str(ts))
        print("per page:"+ str(float(ts)/page_c)+"\n")
        timef.write("test:"+str(ts)+"\n")
        timef.write("per page:"+ str(float(ts)/page_c)+"\n")

class LossHistory(tf.keras.callbacks.Callback):
    '''
    Draw the figure of train
    '''
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        f1 = plt.figure(1)
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')

        f2 = plt.figure(2)
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        plt.plot(iters, self.val_loss[loss_type], 'b', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')

        plt.show()