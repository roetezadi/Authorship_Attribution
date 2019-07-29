import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import DataFuntions
import Model

subset = None

# models hyperparameter
nb_filter = 300
filter_kernels = [3, 4, 5]
dens_output = 256
# output_author = 67
# output_author = 41
# output_author = 23
output_author = 24

maxlen = 140
batch_size = 32
epochs = 20

df, ids_set, ids_list = DataFuntions.read_data()
alphabet, reverse_alphabet, char_size, char_set = DataFuntions.create_persian_alphabets()
# alphabet, reverse_alphabet, char_size, char_set = DataFuntions.create_english_alphabets()
alphabet_size = len(alphabet)

# url = 'dataExcelNormal_mostAuthors.xlsx'
# df = pd.read_excel(url)
# df.columns = ['id','text']
X = (df[['text']])
print X.shape
le = LabelEncoder()
enc = LabelBinarizer()
df['id'] = le.fit_transform(df['id'])
enc.fit(list(df['id']))
Y = enc.transform(list(df['id']))


X_train, X_test, Y_train, Y_test = train_test_split(df[['text']], df['ids_num'], random_state=1)
Y_train = DataFuntions.encode_y(Y_train, output_author)
Y_test = DataFuntions.encode_y(Y_test, output_author)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

# X_train.reshape(22319,1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# print(X_train)
# print(len(y_train))
# print(len(X_test))
# print(len(y_test))
# print maxlen
# print alphabet_size
model = Model.model_cnn(alphabet_size, maxlen, dens_output, output_author, nb_filter, filter_kernels)

file = open('pycharm_report_24author_20epoches_300filters.txt','w')

for e in range(epochs):
    # X_train, X_test, Y_train, Y_test = train_test_split(df['text'], df['ids_num'], random_state=1)
    # Y_train = DataFuntions.encode_y(Y_train, output_author)
    # Y_test = DataFuntions.encode_y(Y_test, output_author)


    xi, yi = DataFuntions.shuffle_matrix(X_train, Y_train)
    xi_test, yi_test = DataFuntions.shuffle_matrix(X_test, Y_test)

    if subset:
        batches = DataFuntions.mini_bactch_data(xi[:subset], yi[:subset], maxlen, alphabet, alphabet_size, char_set, batch_size=batch_size)
    else:
        batches = DataFuntions.mini_bactch_data(xi, yi, maxlen, alphabet, alphabet_size, char_set, batch_size=batch_size)

    test_batches = DataFuntions.mini_bactch_data(xi_test, yi_test, maxlen, alphabet, alphabet_size, char_set, batch_size=batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    for x_train, y_train in batches:
        if len(x_train) == len(y_train):
            f = model.train_on_batch(x_train, y_train)
            loss += f[0]
            loss_avg = loss / step
            accuracy += f[1]
            accuracy_avg = accuracy / step
            if step % 100 == 0:
                print('  Step: {}'.format(step))
                file.write('  Step: '+ str(step) + '\n')
                print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
                file.write('\tLoss: '+ str(loss_avg)+' Accuracy: ' + str(accuracy_avg) + '\n')
            step += 1

    test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1

    for x_test_batch, y_test_batch in test_batches:
        f_ev = model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev[0]
        test_loss_avg = test_loss / test_step
        test_accuracy += f_ev[1]
        test_accuracy_avg = test_accuracy / test_step
        test_step += 1
    print('Epoch {}. Loss: {}. Accuracy: {}\n'.format(e, test_loss_avg, test_accuracy_avg))
    file.write('Epoch ' + str(e) + ' Loss: ' + str(test_loss_avg) + ' Accuracy: ' + str(test_accuracy_avg))

file.close()