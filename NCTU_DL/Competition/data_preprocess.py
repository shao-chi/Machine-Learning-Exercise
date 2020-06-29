import re
from collections import Counter
from ast import literal_eval

import jieba
import pandas as pd
import numpy as np

stopwords = open('../input/baidu-stopwords/baidu_stopwords.txt', 'r').read().split('\n') + [' ', 'NUM']

test_data_path = '../input/dl-course-final-competition/test_data.csv'
train_data_path = '../input/dl-course-final-competition/train_data.csv'
# test_data_path = '../input/dl-competition/test_data_tokens.csv'
# train_data_path = '../input/dl-competition/train_data_tokens.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# label = train_data.drop(columns=['ID', 'title', 'keyword'])
# label = label.drop_duplicates()
# label.to_csv('./data/label.csv', index=False)

# keyword = np.concatenate((train_data['keyword'].values, test_data['keyword'].values))
# keyword = list(set([k for key in keyword for k in key if k != '']))
# keyword.append('<NUM>')
# keyword_dict_path = 'keyword_dict.txt'
# file_ = open(keyword_dict_path, 'w')
# file_.write('\n'.join(keyword))
# file_.close()

# jieba.load_userdict(keyword_dict_path)
jieba.enable_paddle()
train_data = train_data.drop(columns=['label_name'])
train_data['title'] = (train_data['title'] + train_data['keyword'].fillna(',')) \
    .apply(lambda title: re.sub(r'[^\w\s]', '', title)) \
    .apply(lambda title: re.sub(r'\d+', '<NUM>', title)) \
    .apply(lambda title: [word for word in jieba.cut(title, use_paddle=True) \
                                    if word not in stopwords])

test_data['title'] = (test_data['title'] + test_data['keyword'].fillna(','))\
    .apply(lambda title: re.sub(r'[^\w\s]', '', title)) \
    .apply(lambda title: re.sub(r'\d+', '<NUM>', title)) \
    .apply(lambda title: [word for word in jieba.cut(title, use_paddle=True) \
                                    if word not in stopwords])


def build_vocab(l):
    vocab = Counter()
    print(f'Building {l} label vocab..')
    tokens = train_data[train_data['label'] == l]['title'].values
    for token in tokens:
        vocab.update(token)

#     keywords = train_data[train_data['label'] == l]['keyword'].values
#     for key in keywords:
#         vocab.update(key)

    print(f'Original Vocab: {len(vocab.keys())}')
    
    most = [20000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
    vocab = vocab.most_common(most[l])
    # vocab = vocab.most_common()
    vocab = [v[0] for v in vocab]

    return vocab

vocab = list()
for l in range(10):
    vocab_l = build_vocab(l)
    print(f'Label {l}: {len(vocab_l)}')

    vocab.append(vocab_l)

train_data_records = train_data.to_dict('records')
for i in range(len(train_data_records)):
    l = train_data_records[i]['label']
    train_data_records[i]['title'] = [tokens \
                                            for tokens in train_data_records[i]['title'] \
                                                if tokens in vocab[l]]
    train_data_records[i]['keyword'] = [tokens \
                                            for tokens in train_data_records[i]['keyword'] \
                                                if tokens in vocab[l]]

train_data = pd.DataFrame.from_records(train_data_records)

vocab = Counter()
for tokens in train_data['title'].values:
    vocab.update(tokens)
for keyword in train_data['keyword'].values:
    vocab.update(keyword)
for tokens in test_data['title'].values:
    vocab.update(tokens)
for keyword in test_data['keyword'].values:
    vocab.update(keyword)
    
vocab = list(vocab.keys())
print('Vocab: ', len(vocab))

train_data['title'] = train_data['title'] \
    .apply(lambda tokens: [token for token in tokens if token in vocab])
test_data['title'] = test_data['title'] \
    .apply(lambda tokens: [token for token in tokens if token in vocab])
train_data['keyword'] = train_data['keyword'] \
    .apply(lambda keywords: [keyword for keyword in keywords if keyword in vocab])
test_data['keyword'] = test_data['keyword'] \
    .apply(lambda keywords: [keyword for keyword in keywords if keyword in vocab])

# train_data['title'] = train_data['title'] \
#     .apply(lambda tokens: [token for token in tokens])
# test_data['title'] = test_data['title'] \
#     .apply(lambda tokens: [token for token in tokens])
# train_data['keyword'] = train_data['keyword'] \
#     .apply(lambda keywords: [keyword for keyword in keywords])
# test_data['keyword'] = test_data['keyword'] \
#     .apply(lambda keywords: [keyword for keyword in keywords])

train_data.to_csv(f'train_data_{len(vocab)}.csv', index=False)
test_data.to_csv(f'test_data_{len(vocab)}.csv', index=False)
# train_data.to_csv('train_data_full.csv', index=False)
# test_data.to_csv('test_data_full.csv', index=False)