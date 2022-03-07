import tensorflow as tf
import tensorflow.keras.backend as K

def BinaryCrossEntropy(y_true, y_pred, class_weights={'0':3, '1':1}): 
    y_true = tf.cast(y_true, tf.float32)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon()) * class_weights['0']
    term_1 = y_true * K.log(y_pred + K.epsilon()) * class_weights['1']

    return -K.mean(term_0 + term_1, axis=0)