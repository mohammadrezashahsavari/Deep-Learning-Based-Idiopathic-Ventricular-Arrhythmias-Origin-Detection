import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Dense, LSTM, BatchNormalization, AveragePooling1D, ReLU, LeakyReLU, Add, Bidirectional, Concatenate, Dropout, Flatten
from tensorflow.keras.regularizers import l2
import numpy as np



def Shallow_CNN_BiLSTM_Attn_Model(input_shape, n_output_nodes=1):
    inputs = Input(shape=input_shape)
    t = Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape)(inputs)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)

    t = Bidirectional(LSTM(64, return_sequences=True))(t)
    features, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(t)
    state_h = Concatenate()([forward_state_h, backward_state_h])
    state_c = Concatenate()([forward_state_c, backward_state_c])

    context_vector, attention_weights = Attention(64)(features, state_h)

    t = tf.keras.layers.Dense(256)(context_vector)
    t = tf.keras.layers.LeakyReLU(alpha=0.2)(t)
    t = tf.keras.layers.Dropout(0.2)(t)
    t = tf.keras.layers.Dense(256)(t)
    t = tf.keras.layers.LeakyReLU(alpha=0.2)(t)
    t = tf.keras.layers.Dropout(0.2)(t)
    t = tf.keras.layers.Dense(1, activation='sigmoid')(t)

    return Model(inputs, t)


def VGG_Model(input_shape, n_output_nodes=1):
    inputs = Input(shape=input_shape)
    t = Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape)(inputs)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    #t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Flatten()(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.2)(t)
    t = Dropout(0.2)(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.2)(t)
    t = Dropout(0.2)(t)
    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)




def VGG_LSTM_Model(input_shape, n_output_nodes=1):
    inputs = Input(shape=input_shape)
    t = Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape)(inputs)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    #t = MaxPool1D(pool_size=2, strides=2)(t)
    t = LSTM(64, return_sequences=True)(t)
    t = LSTM(64)(t)
    
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)
    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)



def VGG_BiLSTM_Model(input_shape, n_output_nodes=1, l2_lambda=0.001):
    inputs = Input(shape=input_shape)
    t = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    #t = MaxPool1D(pool_size=2, strides=2)(t)

    t = Bidirectional(LSTM(64, return_sequences=True))(t)
    t = Bidirectional(LSTM(64))(t)

    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)

    outputs = Dense(1, activation='sigmoid')(t)

    return Model(inputs, outputs)


def VGG_LSTM_Attn_Model(input_shape, n_output_nodes=1):
    inputs = Input(shape=input_shape)
    t = Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape)(inputs)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    #t = MaxPool1D(pool_size=2, strides=2)(t)

    t = LSTM(64, return_sequences=True)(t)
    features, state_h, state_c = LSTM(64, return_sequences=True, return_state=True)(t)

    context_vector, attention_weights = Attention(64)(features, state_h)

    t = Dense(256)(context_vector)
    t = LeakyReLU(alpha=0.2)(t)
    t = Dropout(0.2)(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.2)(t)
    t = Dropout(0.2)(t)
    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)



def VGG_BiLSTM_Attn_Model(input_shape, n_output_nodes=1):
    inputs = Input(shape=input_shape)
    t = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    #t = MaxPool1D(pool_size=2, strides=2)(t)

    t = Bidirectional(LSTM(64, return_sequences=True))(t)
    features, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(t)
    state_h = Concatenate()([forward_state_h, backward_state_h])
    state_c = Concatenate()([forward_state_c, backward_state_c])

    context_vector, attention_weights = Attention(64)(features, state_h)

    t = Dense(256)(context_vector)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)

    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)


def Resnet_BiLSTM_Attn_Model(input_shape, n_output_nodes=1):
    inputs = Input(shape=input_shape)
    t = residual_block1d(inputs, downsample=True, filters=64, kernel_size=3)
    t = residual_block1d(t, downsample=True, filters=128, kernel_size=3)
    features, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(t)
    state_h = Concatenate()([forward_state_h, backward_state_h])
    state_c = Concatenate()([forward_state_c, backward_state_c])

    context_vector, attention_weights = Attention(64)(features, state_h)

    t = Dense(256)(context_vector)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)

    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)


def Resnet18_LSTM_Attn_Model(num_blocks_list=[2, 2, 2, 2], input_shape=(500, 12), n_output_nodes = 1):
    inputs = Input(shape=input_shape)
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block1d(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    t = AveragePooling1D(4)(t)
    t = LSTM(128, return_sequences=True)(t)
    #Attention
    features, state_h, state_c = LSTM(128, return_sequences=True, return_state=True)(t)
    context_vector, attention_weights = Attention(10)(features, state_h)

    t = Dense(1024, activation='relu')(context_vector)
    t = Dense(1024, activation='relu')(t)
    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block1d(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv1D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
          
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
          
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights







