import numpy as np

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
    for pages in set_count[set_num-1]:
        set_len = pages
        for i in range(set_len):
            feature.append(df[count])
            count += 1
        if set_len != max_set[set_num-1]:
            for i in range(max_set[set_num-1]-set_len):
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
    for pages in set_count[set_num-1]:
        set_len = pages
        for i in range(set_len):
            label.append(df[count])
            count += 1
        if set_len != max_set[set_num-1]:
            for i in range(max_set[set_num-1]-set_len):
                label.append(0)
    return label

def to_train_array_set(df, set_count, set_num, max_set):
    cols = ['Leafnode', 'PTypeSet', 'TypeSet', 'Contentid', 'Pathid', 'Simseqid']
    features = []
    for c in range(len(cols)):
        features.append(np.array(feature_padding_set(df[cols[c]], set_count, set_num, max_set)))
        features[c] = np.expand_dims(features[c], -1)
    
    features = np.concatenate([feature for feature in features], -1)
    features = np.reshape(features, [len(set_count[set_num-1])*max_set[set_num-1], 6])
    
    label = np.array(label_padding_set(df['Label'], set_count, set_num, max_set))
    label = np.reshape(label, [len(set_count[set_num-1])*max_set[set_num-1]])
    return features, label