#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, unicode_literals, print_function, absolute_import
import numpy as np
import tensorflow as tf
import pandas as pd
from crflayer import CRF
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn_crfsuite import metrics
import warnings
import time
import os
import func
import prepare_train_with_set as prepare
import set_func
import glob
warnings.filterwarnings('ignore')


# # Parameter

# In[ ]:


# How many Set
DEBUG = False        # Print element
path_max_len = 30    # padding length
path_emb_size = 5    # embedding size

con_max_len = 50    # padding length
con_emb_size = 5    # embedding size

EPOCHS = 10000        # Train epochs
conv_num = 5        # First cnn filter num
#max_num = 206       # How many nodes should pad
UNTIL_LOSS = 0.001    # When achieve loss then stop
opt = tf.keras.optimizers.Adam(learning_rate=0.0005) # Set learning rate
NO_IMPROVE = 3     # Stop when no improve for epochs
current_path = os.path.join(os.path.expanduser("~"), "jupyter", "Sequence_Labeling_Wrapper_Verification", "data")
data_path = os.path.join(current_path, "data")
set_total = len(glob.glob(os.path.join(data_path, "Set-*")))
print("Set:", set_total)


# # GPU limit

# In[ ]:


def gpu_limit(num):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[num], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# # Tokenizer

# Use tokenizer to convert words to encoding for embedding layer.

# In[ ]:


def tokenizer():
    tokenizer_path = tf.keras.preprocessing.text.Tokenizer(num_words=None)
    tokenizer_content = tf.keras.preprocessing.text.Tokenizer(num_words=None)
    return tokenizer_path, tokenizer_content


# In[ ]:


def process_training(set_total, current_path, tokenizer_path, tokenizer_content, path_len, con_len):
    train_data, Set_dict = prepare.train_file_generate(set_total, current_path)
    test_data = prepare.test_file_generate(current_path)
    max_num_train, max_label_train = func.load_data_num(train_data, True)
    max_num_test = func.load_data_num(test_data, False)
    max_num = max(max_num_train, max_num_test)
    col_set_dict = dict(map(reversed, Set_dict.items()))
    feature_train, word_train, label_train, out_train = func.cnn_process_data(train_data, tokenizer_path, tokenizer_content, path_len, con_len)
    return test_data, feature_train, word_train, label_train, max_label_train, max_num, col_set_dict


# In[ ]:


class EarlyStoppingByLossVal(tf.keras.callbacks.Callback):
    '''
    Early stop when training value less than setting value.
    '''
    def __init__(self, monitor='loss', value=UNTIL_LOSS, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


# # Model

# In[ ]:


def full_model(max_num, max_label):
    '''
    Model definition for our experiments using tensorflow keras.
    '''
    path_input = tf.keras.Input(shape=(path_max_len,), name='Path_emb_input')
    content_input = tf.keras.Input(shape=(con_max_len,), name='Content_emb_input')
    feature_input = tf.keras.Input(shape=(6,), name='Feature_input')
    
    path_emb = tf.keras.layers.Embedding(path_word_size+1, path_emb_size)(path_input)
    content_emb = tf.keras.layers.Embedding(con_word_size+1, con_emb_size)(content_input)
    
    feature = tf. reshape(feature_input, [-1, max_num, 6])
    path_emb = tf.reshape(path_emb, [-1, max_num, path_max_len*path_emb_size, 1])
    content_emb = tf.reshape(content_emb, [-1, max_num, con_max_len*con_emb_size, 1])
    
    path = tf.keras.layers.Conv2D(conv_num, (3, path_max_len*path_emb_size), (1, path_max_len*path_emb_size), padding='same')(path_emb)
    con = tf.keras.layers.Conv2D(conv_num, (3, con_max_len*con_emb_size), (1, con_max_len*con_emb_size), padding='same')(content_emb)
    
    path_emb = tf.reshape(path, [-1, max_num, conv_num])
    content_emb = tf.reshape(con, [-1, max_num, conv_num])
    
    combine = tf.keras.layers.concatenate([feature, path_emb, content_emb], -1)
    
    mlp = combine
    d = tf.reshape(mlp, [-1, 6 + conv_num*2])
    d = tf.keras.layers.Dense(max_label+200, activation='tanh')(d)
    d = tf.keras.layers.Dense(max_label+1, activation='softmax')(d)
    output = d
    model = tf.keras.Model(inputs=[feature_input, path_input, content_input], outputs=output)

    return model

def model_word_only(max_num, max_label):
    '''
    Model definition for our experiments using tensorflow keras.
    '''
    path_input = tf.keras.Input(shape=(path_max_len,), name='Path_emb_input')
    content_input = tf.keras.Input(shape=(con_max_len,), name='Content_emb_input')
    feature_input = tf.keras.Input(shape=(6,), name='Feature_input')
    
    path_emb = tf.keras.layers.Embedding(path_word_size+1, path_emb_size)(path_input)
    content_emb = tf.keras.layers.Embedding(con_word_size+1, con_emb_size)(content_input)
    
    path_emb = tf.reshape(path_emb, [-1, max_num, path_max_len*path_emb_size, 1])
    content_emb = tf.reshape(content_emb, [-1, max_num, con_max_len*con_emb_size, 1])
    
    path = tf.keras.layers.Conv2D(conv_num, (3, path_max_len*path_emb_size), (1, path_max_len*path_emb_size), padding='same')(path_emb)
    con = tf.keras.layers.Conv2D(conv_num, (3, con_max_len*con_emb_size), (1, con_max_len*con_emb_size), padding='same')(content_emb)
    
    path_emb = tf.reshape(path, [-1, max_num, conv_num])
    content_emb = tf.reshape(con, [-1, max_num, conv_num])
    
    combine = tf.keras.layers.concatenate([path_emb, content_emb], -1)
    
    mlp = combine
    d = tf.reshape(mlp, [-1, conv_num*2])
    d = tf.keras.layers.Dense(max_label+200, activation='tanh')(d)
    d = tf.keras.layers.Dense(max_label+1, activation='softmax')(d)
    output = d
    model = tf.keras.Model(inputs=[feature_input, path_input, content_input], outputs=output)

    return model


# # Function define

# In[ ]:


def save_model(model):
    import pickle
    #model.save_weights("./cnn/data/cnn.h5")
    model.save(os.path.join(current_path, "cnn", "data", "model.h5"))
    # saving
    with open(os.path.join(current_path, "cnn", "data", "tokenizer_path.pickle"), "wb") as handle:
        pickle.dump(tokenizer_path, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(current_path, "cnn", "data", "tokenizer_content.pickle"), "wb") as handle:
        pickle.dump(tokenizer_content, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


def load_model():
    import pickle
    model = tf.keras.models.load_model(os.path.join(current_path, "cnn", "data", "model.h5"))
    model.summary()
    # loading
    with open(os.path.join(current_path, "cnn", "data", "tokenizer_path.pickle"), 'rb') as handle:
        tokenizer_path = pickle.load(handle)
    with open(os.path.join(current_path, "cnn", "data", "tokenizer_content.pickle"), 'rb') as handle:
        tokenizer_content = pickle.load(handle)        
    path_word_size = len(tokenizer_path.index_docs)
    con_word_size = len(tokenizer_content.index_docs)
    return model, tokenizer_path, tokenizer_content, path_word_size, con_word_size


# In[ ]:


def get_result(predictions, max_num):
    result = []
    count = 0
    for page in range(int(len(predictions)/max_num)):
        tmp = []
        for node in range(max_num):
            tmp.append(np.argmax(predictions[count]))
            count += 1
        result.append(tmp)
    return result


# In[ ]:


def cnn_process_data(df, tokenizer_path, tokenizer_content, path_max_len, con_max_len):
    '''
    Load the csv file and convert it to np array.
    '''
    max_num, max_label = func.load_data_num(df, True)
    num, index = func.node_num(df['Leafnode'])
    
    num_cols = ['Leafnode', 'PTypeSet', 'TypeSet', 'Contentid', 'Pathid', 'Simseqid']
    features = []
    word_features = []
    tokenizer_path.fit_on_texts(df['Path'])
    tokenizer_content.fit_on_texts(df['Content'])
    path_encoded = tokenizer_path.texts_to_sequences(df['Path'])
    df['Content'] = df['Content'].str.replace('/|\.|\?|:|=|,|<|>|&|@|\+|-|#|~|\|', ' ')
    df['Content'] = df['Content'].astype(str)
    content_encoded = tokenizer_content.texts_to_sequences(df['Content'])
    path_pad = tf.keras.preprocessing.sequence.pad_sequences(path_encoded, path_max_len, padding='post')
    content_pad = tf.keras.preprocessing.sequence.pad_sequences(content_encoded, con_max_len, padding='post')
    
    word_cols = [path_pad, content_pad]
    word_max_len = [path_max_len, con_max_len]
    
    for c in range(len(num_cols)):
        features.append(np.array(func.node_data(df[num_cols[c]], num, max_num)).astype('int32'))
        features[c] = np.expand_dims(features[c], -1)
    
    for c in range(len(word_cols)):
        word_features.append(np.array(func.node_emb(word_cols[c], num, word_max_len[c], max_num)).astype('int32'))
    label_array = np.array(func.label_padding(df['Label'], num, max_num)).astype('int32')
    m_label = df['Label'].max()
    
    feature = np.concatenate([feature for feature in features], -1)
    feature = np.reshape(feature, [features[0].shape[0]*max_num, 6])
    
    word = [np.reshape(word_features[c], [word_features[c].shape[0]*max_num, word_max_len[c]]) for c in range(len(word_cols))]
    feature = feature.astype('float32')
    label_array = label_array.flatten()
    return feature, word, label_array, m_label


# In[ ]:


def emb_padding_set(df, set_count, set_num, pad_len):
    emb = []
    tmp = []
    for i in range(pad_len):
        tmp.append(0)
    count = 0
    for pages in set_count[set_num-1]:
        set_len = pages
        for i in range(set_len):
            emb.append(df[count])
            count += 1
        if set_len != max_set[set_num-1]:
            for i in range(max_set[set_num-1]-set_len):
                emb.append(tmp)
    if DEBUG:
        print(count)
    return emb


# In[ ]:


def cnn_process_set(df, set_count, set_num, max_set, path_max_len, con_max_len):
    num, index = func.node_num(df['Leafnode'])
    cols = ['Leafnode', 'PTypeSet', 'TypeSet', 'Contentid', 'Pathid', 'Simseqid']
    features = []
    word_features = []
    for c in range(len(cols)):
        features.append(np.array(set_func.feature_padding_set(df[cols[c]], set_count, set_num, max_set)).astype('int32'))
        features[c] = np.expand_dims(features[c], -1)
    
    word_cols = ["Path", "Content"]
    word_max_len = [path_max_len, con_max_len]
    for c in range(len(word_cols)):
        word_features.append(np.array(emb_padding_set(df[word_cols[c]], set_count, set_num, word_max_len[c])).astype('int32'))
    
    features = np.concatenate([feature for feature in features], -1)
    features = np.reshape(features, [len(set_count[set_num]) * max_set[set_num], 6])
    
    features = features.astype('float32')
    label = np.array(set_func.label_padding_set(df['Label'], set_count, set_num, max_set)).astype('int32')
    label = label.flatten()
    return features, word_features, label


# In[ ]:


if __name__ == "__main__":
    # How many Set
    model_name = "cnn"
    current_path = os.path.join(os.path.expanduser("~"), "jupyter", "Sequence_Labeling_Wrapper_Verification", "data")
    data_path = os.path.join(current_path, "data")
    set_total = len(glob.glob(os.path.join(data_path, "Set-*")))
    print("Set:", set_total)
    # GPU
    gpu_limit(1)
    
    # Tokenizer
    tokenizer_path, tokenizer_content = tokenizer()
    
    # Process training file
    train_data, Set_dict = prepare.train_file_generate(set_total, current_path)
    test_data = prepare.test_file_generate(current_path)
    max_num_train, max_label_train = func.load_data_num(train_data, True)
    max_num_test = func.load_data_num(test_data, False)
    max_num = max(max_num_train, max_num_test)
    col_set_dict = dict(map(reversed, Set_dict.items()))
    X_train, word_train, y_train, _ = cnn_process_data(train_data, tokenizer_path, 
                                                                         tokenizer_content, path_max_len, 
                                                                         con_max_len)
    
    BATCH_SIZE = max_num      # batch size
    VAL_BATCH_SIZE = max_num  # Validation batch size
    path_word_size = len(tokenizer_path.index_docs)
    con_word_size = len(tokenizer_content.index_docs)
    page_num = int(len(y_train)/max_num)
    
    # Define model
    model = model_word_only(max_num, max_label_train)
    history = func.LossHistory()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    print(model.summary())
    stop_when_no_improve = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0, patience = NO_IMPROVE, restore_best_weights=True)
    until_loss = EarlyStoppingByLossVal(monitor='loss', value=UNTIL_LOSS, verbose=1)
    callbacks = [history, stop_when_no_improve, until_loss]
    
    # Start training
    start = time.time()
    model.fit([X_train, word_train[0], word_train[1]], y_train, epochs=EPOCHS, callbacks=callbacks, use_multiprocessing=True, batch_size=BATCH_SIZE)
    t = time.time()-start
    
    # Graph
    #history.loss_plot('epoch')
    
    # Load test feature
    X_test, word_test, y_test, _ = cnn_process_data(test_data, tokenizer_path, tokenizer_content, path_max_len, con_max_len)
    
    # Start testing
    ts_start = time.time()
    pred = model.predict([X_test, word_test[0], word_test[1]], batch_size=VAL_BATCH_SIZE)
    ts = time.time()-ts_start
    
    # Process output
    result = get_result(pred, max_num)
    col_type = func.get_col_type(current_path)
    Set_data = func.predict_output(set_total, current_path, model_name, col_type, result, max_label_train, col_set_dict)
    set_train_data, set_train_count = set_func.Set_train_file_generate(set_total, current_path, model_name, X_train, word_train, max_num)
    set_test_data, set_test_count = set_func.Set_test_file_generate(set_total, current_path, model_name, Set_data, X_test, word_test, max_num)
    page_c = len(result)
    
    # Process set
    max_num_train = set_func.max_num_set(set_total, set_train_count)
    max_num_test = set_func.max_num_set(set_total, set_test_count)
    max_set = []
    for i in range(len(max_num_train)):
        max_set.append(max(max_num_train[i], max_num_test[i]))
    
    for num in range(set_total):
        set_X_train, set_word_train, set_y_train = cnn_process_set(set_train_data[num], 
                                                                             set_train_count, num, 
                                                                             max_set, path_max_len, 
                                                                             con_max_len)
        max_num = max_set[num]
        max_label = max(set_train_data[num]['Label'])
        BATCH_SIZE = max_num      # batch size
        VAL_BATCH_SIZE = max_num  # Validation batch size
        page_num = int(len(set_X_train)/max_num)
        set_model = model_word_only(max_num, max_label)
        history = func.LossHistory()
        set_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        print(set_model.summary())
        stop_when_no_improve = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0, patience = NO_IMPROVE, restore_best_weights=True)
        until_loss = EarlyStoppingByLossVal(monitor='loss', value=UNTIL_LOSS, verbose=1)
        callbacks = [history, stop_when_no_improve, until_loss]

        # Train
        start = time.time()
        set_model.fit([set_X_train, set_word_train[0], set_word_train[1]], set_y_train, epochs=EPOCHS, callbacks=callbacks, use_multiprocessing=True, batch_size=BATCH_SIZE)
        tst = time.time()-start
        t += tst

        # Load Test file
        set_X_test, set_word_test, set_y_test = cnn_process_set(set_test_data[num], 
                                                                set_test_count, num, 
                                                                max_set, path_max_len, 
                                                                con_max_len)
        
        page_test = int(len(set_X_test) / max_num)

        # Prediction
        ts_start = time.time()
        pred = set_model.predict([set_X_test, set_word_test[0], set_word_test[1]], batch_size=VAL_BATCH_SIZE)
        tsp = time.time()-ts_start
        ts += tsp
        result = get_result(pred, max_num)
        
        # Read Col
        set_col_type = set_func.get_col_type(current_path, num)

        # Output
        set_func.predict_output(current_path, model_name, num, set_col_type, result, max_label, set_X_test, max_num)
    
    # Process time
    func.process_time(current_path, model_name, t, ts, page_c)


# In[ ]:




