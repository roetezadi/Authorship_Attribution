# -*- coding: utf8 -*-
import pandas as pd
import string
import numpy as np
import hazm

normalizer = hazm.Normalizer()


def read_data():
    # df = pd.read_excel('dataExcel1.xlsx', sheet_name='etezadi')
    df = pd.read_excel('dataExcelNormal_mostAuthors.xlsx', sheet_name='etezadi')
    # df = pd.read_csv('dataset.csv')
    df = shuffle_data(df)
    df.columns = ['id', 'text']
    ids_set = set(df['id'])
    ids_list = list(ids_set)
    # print len(ids_set)
    df['ids_num'] = df.loc[:,'id'].map(lambda x: ids_list.index(x))

    return df, ids_set, ids_list

def create_persian_alphabets():
    char = ['ا' ,'ب','پ','ت','ث','ج','چ','ح','خ','د','ذ','ر','ز','ژ','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ک','گ','ل','م','ن','و','ه','ی']
    char += ['۱','۲','۳','۴','۵','۶','۷','۸','۹','۰']
    char += ['؟','ء','آ','اً','هٔ','ة']
    char += (list(string.ascii_lowercase) + list(string.punctuation) + list(string.digits) + ['\n'] + [' '])
    # char += ['U','R','L']
    char = map(lambda x: x.decode('utf8'), char)
    char_size = len(char)
    char_set = set(char)
    # print char_set
    alphabet = {}
    reverse_alphabet = {}
    for idx, c in enumerate(char):
        alphabet[c] = idx
        reverse_alphabet[idx] = c

    return alphabet, reverse_alphabet, char_size, char_set


def create_english_alphabets():
    char = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + ['\n'] + [' '])
    # char = map(lambda x: x.decode('utf8'), char)
    char_size = len(char)
    char_set = set(char)
    # print char_set
    alphabet = {}
    reverse_alphabet = {}
    for idx, c in enumerate(char):
        alphabet[c] = idx
        reverse_alphabet[idx] = c

    return alphabet, reverse_alphabet, char_size, char_set

def encode_data(data_batch_text, maxlen, alphabet, alphabet_size, char_set):
    input_data = np.zeros((len(data_batch_text), maxlen, alphabet_size))
    for idx, text in enumerate(data_batch_text):
        cnt = 0
        text_char_vector = np.zeros((maxlen, alphabet_size))
        # print text
        # text = normalizer.normalize(text)
        # print list(text)
        # text = list(text.decode("utf-8", errors='ignore'))
        text = list(text.lower())
        # print len(text)
        for c in text:
            if cnt >= maxlen:
                pass
            else:
                vect = np.zeros(alphabet_size, dtype=int)
                if c in char_set:
                    # print c
                    index = alphabet[c]
                    # print index
                    vect[index] = 1
                    # print vect
                text_char_vector[cnt, :] = vect
                # print c
                # a = input("type")
                cnt += 1
        input_data[idx, :, :] = text_char_vector

    return input_data

def encode_y(Y, authors):
    Y_data = []
    for y in Y:
        x = np.zeros(authors)
        x[y] = 1
        Y_data.append(x)
    Y_data = np.array(Y_data)
    return Y_data


def mini_bactch_data(X, Y, maxlen, alphabet, alphabet_size, char_set, batch_size=128):
    # print ("here ", len(X))
    # print ("here ", len(Y))
    # print len(Y[16000:16100])
    for i in range(0, len(X), batch_size):
        x_train = X[i:i+batch_size]
        y_train = Y[i:i+batch_size]
        # print(i, "    ", i+batch_size)
        # print len(x_train)
        # print len(y_train)
        # print (len(x_train.iloc[0]))
        input_data = encode_data(x_train, maxlen, alphabet, alphabet_size, char_set)
        # print input_data[0][0]
        # print "---------------"
        yield input_data, y_train


def shuffle_data(df):
    shuffled_df = df.reindex(np.random.permutation(df.index))
    return shuffled_df

def shuffle_matrix(x, y):
    print (x.shape, y.shape)
    stacked = np.hstack((np.matrix(x), y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi

# read_data()
# data = pd.DataFrame({ 'x':[1,2,3,4,5], 'y':[4,5,67,8,9]})
# print(data)
# print(shuffle_data(data)))
# read_data()
# a = "سلام خوبی؟؟"
# print a[2:4]
# # k = 0
# # for i in range(len(a)):
# #     if i+1 <= len(a) and i+2 <= len(a):
# #         if a[k:k+1] == ' ':
# #             print a[k:k+1]
# #             k+=1
# #         else:
# #             print a[k:k+2]
# #             k+=2
# c = list(a)
# new = []
# k = 0
# new = a.decode("utf-8")
# for i in new:
#     print i
# print new