from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv1D, Dense, Dropout, GlobalAveragePooling1D,
                          Input, Lambda, MaxPooling1D, Softmax)
from keras import regularizers
from keras.models import Model

def FriDNN(input_size):

    reg_input = Input(shape=(input_size, 1))

    conv1 = Conv1D(48, 32, strides=6, padding='same', kernel_regularizer=regularizers.l2(0.0001), name='conv1_1')(
        reg_input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    stage2 = stage_layer_with_shortcuts_pad(act1, 48, 3, 2)

    global_avg = GlobalAveragePooling1D()(stage2)

    final_output = Dense(3, activation='softmax')(global_avg)

    return Model(inputs=reg_input, outputs=final_output)


def stage_layer_with_shortcuts_pad(input_layer, dimension, stride_factor, stage_num):

    conv2 = Conv1D(dimension, 8, strides=stride_factor, padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   name='conv{}_1'.format(stage_num))(input_layer)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    conv3 = Conv1D(dimension, 8, padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   name='conv{}_2'.format(stage_num))(act2)
    bn3 = BatchNormalization()(conv3)

    input_layer = MaxPooling1D(1, stride_factor, padding='same')(input_layer)
    input_feature_dim = int(input_layer.shape[2])

    if input_feature_dim != int(bn3.shape[2]):

        input_layer = Lambda(zero_pad)(input_layer)

        if int(input_layer.shape[2]) - int(bn3.shape[2]) == 32:
            input_layer = Lambda(get_sliced32)(input_layer)
        elif int(input_layer.shape[2]) - int(bn3.shape[2]) == 48:
            input_layer = Lambda(get_sliced48)(input_layer)
        elif int(input_layer.shape[2]) - int(bn3.shape[2]) == 64:
            input_layer = Lambda(get_sliced64)(input_layer)

    act3 = Activation('relu')(Add()([bn3, input_layer]))

    conv4 = Conv1D(dimension, 8, padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   name='conv{}_3'.format(stage_num))(act3)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)

    conv5 = Conv1D(dimension, 8, padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   name='conv{}_4'.format(stage_num))(act4)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(Add()([bn5, act3]))

    conv6 = Conv1D(dimension, 8, padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   name='conv{}_5'.format(stage_num))(act5)
    bn6 = BatchNormalization()(conv6)
    act6 = Activation('relu')(bn6)

    conv7 = Conv1D(dimension, 8, padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   name='conv{}_6'.format(stage_num))(act6)
    bn7 = BatchNormalization()(conv7)
    act7 = Activation('relu')(Add()([bn7, act5]))

    return act7


def zero_pad(x___):
    y = K.zeros_like(x___)
    y = K.concatenate([x___, y], axis=-1)
    return y


def get_sliced32(x__):
    x__ = x__[:, :, :-32]
    return x__


def get_sliced48(x__):
    x__ = x__[:, :, :-48]
    return x__


def get_sliced64(x__):
    x__ = x__[:, :, :-64]
    return x__
