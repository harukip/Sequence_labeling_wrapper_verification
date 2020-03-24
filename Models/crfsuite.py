#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
import sklearn_crfsuite
import time
from tqdm import tqdm
import pandas as pd
import prepare_train_with_set as prepare
import os
import func
import set_func
import glob


# In[ ]:


def CRFSuite_process_data(df, max_num, max_label):
    '''
    Load the csv file and convert it to numpy array for train and test.
    '''
    num, index = func.node_num(df['Leafnode'])
    cols = ['Leafnode', 'PTypeSet', 'TypeSet', 'Contentid', 'Pathid', 'Simseqid']
    features = []
    for c in range(len(cols)):
        features.append(np.array(func.node_data(df[cols[c]], num, max_num)))
        features[c] = np.expand_dims(features[c], -1)
    
    feature = np.concatenate([feature for feature in features], -1)
    feature = np.reshape(feature, [features[0].shape[0]*max_num, 6])
    
    label_array = np.array(func.label_padding(df['Label'], num, max_num)).astype('int32')
    m_label = df['Label'].max()
    label_array = label_array.flatten()
    return feature, label_array, m_label

def CRFSuite_process_set(df, set_count, set_num, max_set):
    cols = ['Leafnode', 'PTypeSet', 'TypeSet', 'Contentid', 'Pathid', 'Simseqid']
    features = []
    for c in range(len(cols)):
        features.append(np.array(set_func.feature_padding_set(df[cols[c]], set_count, set_num, max_set)))
        features[c] = np.expand_dims(features[c], -1)
    
    features = np.concatenate([feature for feature in features], -1)
    features = np.reshape(features, [len(set_count[set_num]) * max_set[set_num], 6])
    
    label = np.array(set_func.label_padding_set(df['Label'], set_count, set_num, max_set))
    label = np.reshape(label, [len(set_count[set_num]) * max_set[set_num]])
    return features, label


# In[ ]:


def model():
    return sklearn_crfsuite.CRF(
        c1=0.1,
        max_iterations=50
    )


# In[ ]:


if __name__ == "__main__":
    # How many Set
    model_name = "crfsuite"
    current_path = os.path.join(os.path.expanduser("~"), "jupyter", "Sequence_Labeling_Wrapper_Verification", "data")
    data_path = os.path.join(current_path, "data")
    set_total = len(glob.glob(os.path.join(data_path, "Set-*")))
    print("Set:", set_total)
    # Process training file
    train_data, Set_dict = prepare.train_file_generate(set_total, current_path)
    test_data = prepare.test_file_generate(current_path)
    max_num_train, max_label_train = func.load_data_num(train_data, True)
    max_num_test = func.load_data_num(test_data, False)
    max_num = max(max_num_train, max_num_test)
    col_set_dict = dict(map(reversed, Set_dict.items()))
    feature_train, label_train, out_train = CRFSuite_process_data(train_data, max_num_train, max_label_train)
    feature_train = feature_train.tolist()
    label_train = label_train.tolist()
    X_train = [func.sent2features(feature_train)]
    y_train = [func.sent2labels(label_train)]
    
    # Define model
    crf = model()
    
    # Start training
    start = time.time()
    crf.fit(X_train, y_train)
    t = time.time() - start
    
    # Process testing file
    feature_test, label_test, _ = CRFSuite_process_data(test_data, max_num, max_label_train)
    feature_test = feature_test.tolist()
    label_test = label_test.tolist()
    X_test = [func.sent2features(feature_test)]
    page_test = int(len(feature_test)/max_num)
    
    # Start testing
    ts_start = time.time()
    y_pred = crf.predict(X_test)
    ts = time.time() - ts_start
    
    # Process output
    y_pred = np.array(y_pred)
    result = np.reshape(y_pred, [page_test, max_num])
    result = result.astype(np.int32)
    
    col_type = func.get_col_type(current_path)
    Set_data = func.predict_output(set_total, current_path, model_name, col_type, result, max_label_train, col_set_dict)
    set_train_data, set_train_count = set_func.Set_train_file_generate(set_total, current_path, model_name, feature_train, "", max_num)
    set_test_data, set_test_count = set_func.Set_test_file_generate(set_total, current_path, model_name, Set_data, feature_test, "", max_num)
    page_c = len(result)
    
    # Process set
    if set_total > 0:
        max_num_train = set_func.max_num_set(set_total, set_train_count)
        max_num_test = set_func.max_num_set(set_total, set_test_count)
        max_set = []
        for i in range(len(max_num_train)):
            max_set.append(max(max_num_train[i], max_num_test[i]))

        for num in range(set_total):
            set_feature_train, set_label_train = CRFSuite_process_set(set_train_data[num], set_train_count, num, max_set)
            max_num = max_set[num]
            max_label = max(set_train_data[num]['Label'])
            page_num = int(len(set_feature_train)/max_num)
            crf = model()
            set_feature_train = set_feature_train.tolist()
            set_label_train = set_label_train.tolist()
            X_train = [func.sent2features(set_feature_train)]
            y_train = [func.sent2labels(set_label_train)]

            # Train
            start = time.time()
            crf.fit(X_train, y_train)
            t += time.time()-start

            # Load Test file
            set_feature_test, set_label_test = CRFSuite_process_set(set_test_data[num], set_test_count, num, max_set)
            set_feature_test = set_feature_test.tolist()
            set_label_test = set_label_test.tolist()
            X_test = [func.sent2features(set_feature_test)]
            page_test = int(len(set_feature_test) / max_num)

            # Prediction
            ts_start = time.time()
            y_pred = crf.predict(X_test)
            ts += time.time() - ts_start

            y_pred = np.array(y_pred)
            result = np.reshape(y_pred, [page_test, max_num])
            result = result.astype(np.int64)
            # Read Col
            col_type = set_func.get_col_type(current_path, num)

            # Output
            set_func.predict_output(current_path, model_name, num, col_type, result, max_label, set_feature_test, max_num)
            
    # Process time
    func.process_time(current_path, model_name, t, ts, page_c)


# In[ ]:




