from keras.models import Model
from keras.optimizers import sgd
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D

def model_cnn(alphabet_size, maxlen, output_dim, output_author, nb_filter, filter_kernels):

    inputs = Input(shape=(maxlen, alphabet_size), name='input', dtype='float32')
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu', name='conv_layer1',
                         input_shape=(maxlen, alphabet_size))(inputs)
    conv = MaxPooling1D(pool_length=2)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid',name='conv_layer2', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=2)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],name='conv_layer3',
                          border_mode='valid', activation='relu')(conv1)
    conv3 = Flatten()(conv2)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(output_dim, activation='relu')(conv3))
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    pred = Dense(output_author, activation='softmax', name='output')(z)

    model = Model(input=inputs, output=pred)

    _sgd = sgd(lr=0.05, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=_sgd,
                  metrics=['accuracy'])

    # model.summary()

    return model
